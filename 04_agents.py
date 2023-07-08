from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
import os

load_dotenv()

OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Using an agent with LLM

llm = AzureOpenAI(
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    temperature=0.1
)

from langchain.chat_models import AzureChatOpenAI
chat = AzureChatOpenAI(
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    temperature=0
)

tools = load_tools(["openweathermap-api", "llm-math"], llm=llm)

agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

# Now let's test it out!
agent.run("What is the temperature in Helsinki right now? What is that number doubled?")