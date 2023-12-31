import os
import glob
import textwrap
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from huggingface_hub import hf_hub_download


from rich import console
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from functools import reduce
from itertools import chain
from datetime import datetime

# Define Hugging Face Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_OWN_HUGGING_FACE_API_KEY"

# Document Loader
loader = TextLoader('KS-all-info_rev1.txt')
documents = loader.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings()

#Create the vectorized db
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
db = FAISS.from_documents(docs, embeddings)

template = """ 
You are going to be my assistant.
Please try to give me the most beneficial answers to my
question with reasoning for why they are correct.

Question: {input} 
Answer: 
"""
console = Console()
console.print("[bold yellow]Preparing the DQA Model...")

prompt = PromptTemplate(
    template=template, 
    input_variables=["input"]
)

llm=HuggingFaceHub(
    # repo_id="MBZUAI/LaMini-Flan-T5-783M", 
    repo_id="declare-lab/flan-alpaca-large",
    model_kwargs={"temperature":0, "max_length":512}
)
chain = load_qa_chain(llm, chain_type="stuff")

while True:
    query = console.input("Ask DQA: (enter q for quit): ")
    if "q" == query.lower():
        console.print("[red blink]Exiting...!")
        break
    start = datetime.now()
    console.print("[red blink]Executing...")
    console.print(f"[grey78]Generating answer to your question:[grey78] [green_yellow]{query}")
    #query = 'Write a travel blog about a 3-day trip to The Philippines'
    docs = db.similarity_search(query)
    
    response = chain.run(input_documents=docs, question=query)
    wrapped_text = textwrap.fill(response, 100)
    console.print(Panel(wrapped_text, title="DQA Reply", title_align="center"))
    stop = datetime.now()
    elapsed = stop - start
    console.rule(f"Report Generated in {elapsed}")
    console.print(f"DQA @ {datetime.now().ctime()}")