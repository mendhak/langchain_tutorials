from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# This is an LLMChain to write a synopsis given a title of a play.
llm = AzureOpenAI(temperature=.7, deployment_name=DEPLOYMENT_NAME)
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a review of a play given a synopsis.
llm = AzureOpenAI(temperature=.7, deployment_name=DEPLOYMENT_NAME)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

review = overall_chain.run("Tragedy at sunset on the beach")
print(review)