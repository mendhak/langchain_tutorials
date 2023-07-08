import os
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain import ConversationChain
from langchain.prompts import ( ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# llm = AzureOpenAI(temperature=0, deployment_name=DEPLOYMENT_NAME,)
# conversation = ConversationChain(llm=llm, verbose=True)
# print(conversation.run("Hi there!"))
# print(conversation.run("I'm doing well! Just having a conversation with an AI."))


prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(""" The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI doe snot know the answer to a question, it truthfully says it does not know."""),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = AzureChatOpenAI(temperature=0, deployment_name=DEPLOYMENT_NAME,)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

print(conversation.predict(input="Hi there!"))
print(conversation.predict(input="Tell me about yourself"))
print(conversation.predict(input="By whom?")) # After it says it was created using NLP. The 'by whom' is in that context. 


