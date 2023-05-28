"""
DATE: 2023/5/28
AUTHOR: ZLYANG
CONTACT: zhlyang95@hotmail.com
"""

### define tools ###

import requests
import io
import base64
import os
from PIL import Image
from typing import Optional

from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
import re, random
from hashlib import md5


# translation from baidu api
def translate_to_en(text, appid, appkey):
    url = "http://api.fanyi.baidu.com/api/trans/vip/translate"

    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    params = {
        "appid": appid,
        "q": text,
        "from": "zh",
        "to": "en",
        "salt": salt,
        "sign": sign
    }
    r = requests.post(url, params=params, headers=headers).json()
    result = r["trans_result"][0]["dst"]
    return result


# base api tool #
class APITool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_api(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_api(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")


# search tool #
class SearchTool(APITool):
    llm: BaseLanguageModel

    # tool description
    name = "搜索问答"
    description = "根据用户问题搜索最新的结果，并返回Json格式的结果"

    # search params
    google_api_key: str
    google_cse_id: str
    url = "https://www.googleapis.com/customsearch/v1"
    top_k = 5

    # QA params
    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    如果该文本中没有相关内容可以回答问题，请直接回复：“抱歉，该问题需要更多上下文信息。”
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_api(self, query) -> str:
        self.get_llm_chain()
        context = self.get_search_result(query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_search_result(self, query):
        data = {"key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "lr": "lang_zh-CN"}
        results = requests.get(self.url, params=data).json()
        results = results.get("items", [])[:self.top_k]
        snippets = []
        if len(results) == 0:
            return("No Search Result was found")
        for result in results:
            text = ""
            if "title" in result:
                text += result["title"] + "。"
            if "snippet" in result:
                text += result["snippet"]
            snippets.append(text)
        return("\n\n".join(snippets))

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)


# draw tool #
class DrawTool(APITool):
    # tool description
    name = "绘画"
    description = "根据用户描述调用api画图"

    # stable diffusion api
    baidu_appid: str
    baidu_appkey: str
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

    def _call_api(self, query) -> str:
        img_path = self.get_response(query)
        return f"图片保存在{img_path}下"

    def get_response(self, query):
        draw_prompt = translate_to_en(query, self.baidu_appid, self.baidu_appkey)
        draw_prompt += ",traditional chinese ink painting,peaceful"
        negative_prompt = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, age spot, glans, (watermark:2)"
        img_path = "output/images/"

        payload = {
            "prompt": draw_prompt,
            "steps": 30,
            "width": 640,
            "height": 1024,
            "negative_prompt": negative_prompt,
            "sampler_index": "DPM++ SDE Karras",
            "cfg_scale": 3.5
        }
        response = requests.post(url=self.url, json=payload)
        r = response.json()
        for i, img in enumerate(r["images"]):
            image = Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
            image.save(img_path + f"output_{i}.png")
        return img_path


# audio tool #
class AudioTool(APITool):
    # tool description
    name = "语音"
    description = "根据用户的输入描述，将一定格式下的文本内容转成语音"

    # Chinese-FastSpeech2
    url = "http://127.0.0.1:5876/TextToSpeech"

    def _call_api(self, query) -> str:
        res = re.search(r"<.+>", query)
        if res:
            speech_text = res.group()[1:-1]
            save_path = "./output/audios/"
            save_path = os.path.abspath(save_path)

            payload = {
                "text": speech_text,
                "save_path": save_path
            }
            res = requests.post(self.url, payload).json()
            audio_path = res["result"]
            return f"图片保存在{audio_path}下"
        else:
            print("转语音的文本要按格式输入，用<>括起来！")

