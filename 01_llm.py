from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
import os

load_dotenv()

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Get prediction from language model

llm = AzureOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0.9
)

print(llm.predict("What would be a good company name for a company that makes colorful socks?"))


