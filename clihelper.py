import pickle
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import sys

load_dotenv()

input_request="Show my current path"

# If arguments are passed in, concatenate them together in a string
if len(sys.argv) > 1:
    input_request = " ".join(sys.argv[1:])


OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

print("Loaded OPENAI credentials")

model = AzureChatOpenAI(
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    temperature=0.3
)

print("Loaded Azure Model")


template = """Assistant is a large language model trained by OpenAI.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

chain_memory=ConversationBufferWindowMemory(k=2)
resuming_conversation = False

if os.path.isfile('history.pickle'):
    with open('history.pickle', 'rb') as handle:
        chain_memory = pickle.load(handle)
    resuming_conversation = True
    

chatgpt_chain = LLMChain(
    llm=model, 
    prompt=prompt, 
    verbose=True, 
    memory=chain_memory,
)

if not resuming_conversation:
    output = chatgpt_chain.predict(human_input="I want you to act as a Linux helper. I will describe what I want to do, and you will reply with a Linux command to accomplish that task. I want you to only reply with the Linux command inside one unique code block, and nothing else. Do not write explanations. Only output the command in a unique code block. My first request is, how do I show my current path.")

    # Simply ignoring the first output

output = chatgpt_chain.predict(human_input=input_request)
print(output)

with open('history.pickle', 'wb') as handle:
    pickle.dump(chatgpt_chain.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)


