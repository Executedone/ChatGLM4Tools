"""
DATE: 2023/5/28
AUTHOR: ZLYANG
CONTACT: zhlyang95@hotmail.com
"""

# run example

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