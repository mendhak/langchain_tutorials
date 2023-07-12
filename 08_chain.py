import os
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain

load_dotenv()
# DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

llm = AzureChatOpenAI(temperature=0.9, deployment_name=DEPLOYMENT_NAME)
prompt_template = "Tell me a {adjective} joke"
llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

print(llm_chain.run({"adjective": "silly"}))


