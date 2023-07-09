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


DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

model = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0.3
)

template = "You are a helpful assistant that outputs example Linux commands.I will describe what I want to do, and you will reply with a Linux command to accomplish that task. I want you to only reply with the Linux command inside one unique code block, and nothing else. Do not write explanations. Only output the command in a unique code block."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human_1 = HumanMessagePromptTemplate.from_template("List files in the current directory")
example_ai_1 = AIMessagePromptTemplate.from_template("```\nls\n```")
example_human_2 = HumanMessagePromptTemplate.from_template("Push my git branch up")
example_ai_2 = AIMessagePromptTemplate.from_template("```\ngit push origin <branchname>\n```")
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_human_1, example_ai_1, example_human_2, example_ai_2, human_message_prompt]
)
chain = LLMChain(llm=model, prompt=chat_prompt)
# get a chat completion from the formatted messages
print(chain.run("git remove executable bit from file"))