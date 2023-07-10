import os
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.llms import AzureOpenAI
from langchain import ConversationChain
from langchain.prompts import ( ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate)
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = AzureChatOpenAI(temperature=0, deployment_name=DEPLOYMENT_NAME)

_input = prompt.format(subject="pasta types")
output = model.predict(_input)

print(output_parser.parse(output))