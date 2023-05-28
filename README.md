# ChatGLM4Tools
基于LangChain框架，以ChatGLM-6B为语言模型，实现工具调用

chatglm-6B for tools application using langchain

## 介绍
本项目通过langchain整合chatglm-6B，实现了多工具的调用demo，项目具体情况如下：

1、所有工具以api的方式进行调用；

2、目前实现了3类工具：**搜索问答、绘画、语音**；
- **搜索问答工具**调用google搜索，再利用模型对搜索结果进行信息整合，回答用户的问题。调用该工具需要google api_ley和google cse_id，申请及操作方法见 [Programmable Search Engine](https://developers.google.com/custom-search/v1/using_rest?hl=zh-cn)。结合搜索的回答可以有效缓解模型的幻觉问题。

- **绘画工具**使用的是[Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)，需提前在本地部署并启动，启动时命令行加上--api，可以起接口调用。调用参数可参考[sd api](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)。
  此外由于sd主要是英文作为prompt，而用户的输入为中文，所以在这个工具里面调用了百度翻译，使用时需要百度的appid和appkey，申请方法见[百度开发者注册](http://api.fanyi.baidu.com/doc/12)。当然您也可以替换成自己的翻译工具，自由度很高。
  
- **语音工具**是将一段文本转成语音文件，使用的是项目[Chinese-FastSpeech2](https://github.com/Executedone/Chinese-FastSpeech2)，在使用时要将需要转语音的文本用<>括起来。具体可参考下面示例。

3、生成的图片及语音文件在output目录下对应的文件夹中。

4、model文件夹下放chatglm-6B相关的模型文件及python文件

5、chatglm-6B参数量不算很大，模型本身有不少局限性，因此实现调用工具主要是通过模型对用户的问题进行意图识别，先判断用户的意图再进行工具的调用。由于目前只实现了三类工具，所以只支持这三类意图。如果想增加或自定义，可通过增加意图类别及对应的工具即可。新增工具可通过继承APITool基类实现扩展。

## 示例
参考run.py中的样例。
```python
from langchain.agents import AgentExecutor
from llm import ChatGLM
from tools import SearchTool, DrawTool, AudioTool
from agent import IntentAgent

# google search api ley
GOOGLE_API_KEY = "xxx"
GOOGLE_CSE_ID = "xxx"

# baidu translate api key
BAIDU_APPID = "xxx"
BAIDU_APPKEY = "xxx"

llm = ChatGLM(model_path="<absolute model path, including model weights and related chatgim-6B python scripts>")
llm.load_model()

tools = [SearchTool(llm=llm, google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID),
         DrawTool(baidu_appid=BAIDU_APPID, baidu_appkey=BAIDU_APPKEY),
         AudioTool()]

agent = IntentAgent(tools=tools, llm=llm)
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
agent_exec.run("登幽州台歌的作者是谁？")
# > Entering new AgentExecutor chain...
# 意图类别：搜索问答《登幽州台歌》的作者是唐代诗人陈子昂(661年-702年)。这首诗通过描写登楼远眺，凭今吊古所引起的无限感慨，抒发了诗人抑郁已久的悲愤之情，深刻地揭示了封建社会中那些怀才不遇的知识阶层的苦闷和无奈。
# > Finished chain.

## 原模型这个问题回答如下：
## 《登幽州台歌》的作者是唐代诗人王之涣。这首诗是王之涣的代表作之一，描述了诗人登上幽州台后俯瞰远方景色，感叹世事变迁和个人命运的感慨。这首诗语言简练，意境深远，被誉为唐代诗歌中的经典之作。
## 原模型在某些知识问答上效果不佳，可通过搜索来补充。

agent_exec.run("画一幅画，内容为一个英姿飒爽的女人骑着一匹马")
# > Entering new AgentExecutor chain...
# 意图类别：绘画图片保存在output/images/下
# > Finished chain.

agent_exec.run("把下面的文本生成语音，内容为<一个英姿飒爽的女人骑着一匹马>")
# > Entering new AgentExecutor chain...
# 意图类别：语音图片保存在output/audios/mp.wav下
# > Finished chain.
```

## 局限性
1、一轮对话中仅支持单个工具的调用；

2、目前只做了单轮的调用，后续可以把历史对话放入，支持多轮。
