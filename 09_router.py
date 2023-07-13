import os
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain

load_dotenv()
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

llm = AzureChatOpenAI(temperature=0.9, deployment_name=DEPLOYMENT_NAME)

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

destination_chains={}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")    

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)

print(chain.run("What is the force of gravity in the Earth's atmosphere?"))
print(chain.run("What is the name of the type of cloud that rains"))

# Router Chains but with Embeddings to find similarities - requires COHERE API keys etc

# from langchain.chains.router.embedding_router import EmbeddingRouterChain
# from langchain.embeddings import CohereEmbeddings
# from langchain.vectorstores import Chroma

# names_and_descriptions = [
#     ("physics", ["for questions about physics"]),
#     ("math", ["for questions about math"]),
# ]

# router_chain = EmbeddingRouterChain.from_names_and_descriptions(
#     names_and_descriptions, Chroma, CohereEmbeddings(), routing_keys=["input"]
# )

# chain = MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains=destination_chains,
#     default_chain=default_chain,
#     verbose=True,
# )

# print(chain.run("What is black body radiation?"))