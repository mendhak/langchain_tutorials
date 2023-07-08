import pickle
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import sys

load_dotenv()

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

from langchain.agents import AgentType
from langchain.agents import initialize_agent

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser, # A synchronous browser is available, though it isn't compatible with jupyter.
)

# This import is required only for jupyter notebooks, since they have their own eventloop
import nest_asyncio
nest_asyncio.apply()

# async_browser = create_async_playwright_browser()
sync_browser = create_sync_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = browser_toolkit.get_tools()

llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0.3
)
agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent_chain.run(input="Are https://onlyrss.org/posts/the-air-india-order.html and summarize the text, please.")
print(response)