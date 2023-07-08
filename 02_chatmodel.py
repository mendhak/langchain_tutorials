from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
import os
from langchain.schema import ( AIMessage, HumanMessage, SystemMessage )

load_dotenv()

OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Get prediction from language model

chat = AzureChatOpenAI(
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    temperature=0
)

output = chat.predict_messages([HumanMessage(content="Translate this sentence from English to Spanish. My spoon is too big.")])

print(output)

print(chat.predict("Translate this sentence from English to Spanish. My spoon is too big."))