# TODO: This keeps falling into explanations. Try a "FEW SHOTS" example?

import pickle
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import sys

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

input_request="Show my current path"

# If arguments are passed in, concatenate them together in a string
if len(sys.argv) > 1:
    input_request = " ".join(sys.argv[1:])

chain_memory=ConversationBufferWindowMemory(k=2)
resuming_conversation = False

if os.path.isfile('history.clihelper.pickle'):
    with open('history.clihelper.pickle', 'rb') as handle:
        chain_memory = pickle.load(handle)
    resuming_conversation = True

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

model = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0.3,
    verbose=True
)


template = "You are a helpful assistant that outputs example Linux commands.I will describe what I want to do, and you will reply with a Linux command to accomplish that task. I want you to only reply with the Linux Bash command inside one unique code block, and nothing else. Do not write explanations. Only output the command in a unique code block. If you don't have a Linux command to respond with, say you don't know, in an echo command"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human_1 = HumanMessagePromptTemplate.from_template("List files in the current directory")
example_ai_1 = AIMessagePromptTemplate.from_template("```\nls\n```")
example_human_2 = HumanMessagePromptTemplate.from_template("Push my git branch up")
example_ai_2 = AIMessagePromptTemplate.from_template("```\ngit push origin <branchname>\n```")
example_human_3 = HumanMessagePromptTemplate.from_template("What is your name?")
example_ai_3 = AIMessagePromptTemplate.from_template("```\necho Sorry, I don't know a bash command for that.\n```")
human_template = "{history}\n{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_human_1, example_ai_1, example_human_2, example_ai_2, example_human_3, example_ai_3, human_message_prompt]
)

chain = LLMChain(llm=model, prompt=chat_prompt, memory=chain_memory, verbose=True)

# get a chat completion from the formatted messages
print(chain.run(input_request))

with open('history.clihelper.pickle', 'wb') as handle:
    pickle.dump(chain.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
