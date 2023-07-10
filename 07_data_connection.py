import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()
# DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# raw_documents = TextLoader('./sampletext.txt').load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)

# embeddings = OpenAIEmbeddings(deployment="ada-002-deployment", chunk_size=1)
# db = FAISS.from_documents(documents, embeddings)

# query = "What is the problem with Ruby?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

# Now try loading HTML docs using Playwright
from langchain.document_loaders import PlaywrightURLLoader

urls = [
    
    "https://code.mendhak.com/escaping-jekyll-to-eleventy/",
    "https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/",
    "https://www.cirium.com/thoughtcloud/analysis-china-slower-post-pandemic-aviation-recovery/"
]

loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])

data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(deployment="ada-002-deployment", chunk_size=1)
db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()

question = "How is China's recovery going?"
docs = retriever.get_relevant_documents(question)

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm = AzureChatOpenAI(temperature=0, deployment_name=DEPLOYMENT_NAME), 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents = True,
    verbose = True
    )

result = chain({"question": question})
print(result['answer'])
