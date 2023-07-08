from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
import os

load_dotenv()

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Using an agent with LLM

llm = AzureOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0.1
)

from langchain.chat_models import AzureChatOpenAI
chat = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0
)

tools = load_tools(["openweathermap-api", "llm-math"], llm=llm)

agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

# Now let's test it out!
agent.run("What is the temperature in Helsinki right now? What is that number doubled?")