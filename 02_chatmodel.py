from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
import os
from langchain.schema import ( AIMessage, HumanMessage, SystemMessage )

load_dotenv()

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Get prediction from language model

chat = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0
)

output = chat.predict_messages([HumanMessage(content="Translate this sentence from English to Spanish. My spoon is too big.")])

print(output)

print(chat.predict("Translate this sentence from English to Spanish. My spoon is too big."))