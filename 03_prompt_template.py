from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import AzureChatOpenAI
import os
from langchain.prompts.chat import ( ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)


load_dotenv()

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Set up the templates

chat = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    temperature=0
)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#chat_prompt.format_messages(input_language="English", output_language="Spanish", text="My spoon is too big.")

# Use chains to combine model and prompt template
chain = LLMChain(llm=chat, prompt=chat_prompt)

output = chain.run(input_language="English", output_language="Spanish", text="My spoon is too big.", num_samples=1)
print(output)