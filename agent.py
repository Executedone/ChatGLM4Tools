"""
DATE: 2023/5/7
AUTHOR: ZLYANG
CONTACT: zhlyang95@hotmail.com
"""


### define agent ###

from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel


class IntentAgent(BaseSingleActionAgent):
    tools: List
    llm: BaseLanguageModel
    intent_template: str = """
    现在有一些意图，类别为{intents}，你的任务是理解用户问题的意图，并判断该问题属于哪一类意图。
    回复的意图类别必须在提供的类别中，并且必须按格式回复：“意图类别：<>”。
    
    举例：
    问题：今天的天气怎么样？
    意图类别：搜索问答
    
    问题：画一幅画，内容为山水鸟虫。
    意图类别：绘画
    
    问题：将下面的文字转成语音：<文本>
    意图类别：语音

    问题：“{query}”
    """
    prompt = PromptTemplate.from_template(intent_template)
    llm_chain: LLMChain = None

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def choose_tools(self, query) -> List[str]:
        self.get_llm_chain()
        tool_names = [tool.name for tool in self.tools]
        resp = self.llm_chain.predict(intents=tool_names, query=query)
        select_tools = [(name, resp.index(name)) for name in tool_names if name in resp]
        select_tools.sort(key=lambda x:x[1])
        return [x[0] for x in select_tools]

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        # only for single tool
        tool_name = self.choose_tools(kwargs["input"])[0]
        return AgentAction(tool=tool_name, tool_input=kwargs["input"], log="")

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        raise NotImplementedError("IntentAgent does not support async")