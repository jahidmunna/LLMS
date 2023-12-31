import os
import textwrap
from glob import glob
# from langchain import PromptTemplate, HuggingFaceHub, LLMChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.document_loaders import TextLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from huggingface_hub import hf_hub_download

from langchain.document_loaders import TextLoader  #for textfiles
from langchain.text_splitter import CharacterTextSplitter #text splitter
from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS  #facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader  #load pdf
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader  #load urls into docoument-loader


from rich import console
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from functools import reduce
from itertools import chain
from datetime import datetime

# Define Hugging Face Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_OWN_HUGGING_FACE_API_KEY"

# Load Pdf files and index
DIR = 'docs'
loaders = [UnstructuredPDFLoader(path) for path in glob(f'{DIR}/*.pdf')]
index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        ).from_loaders(loaders)        
        
console = Console()
console.print("[bold yellow]Preparing the DQA Model...")

llm=HuggingFaceHub(
    # repo_id="MBZUAI/LaMini-Flan-T5-783M", 
    repo_id="declare-lab/flan-alpaca-large",
    model_kwargs={"temperature":0, "max_length":512}
)
chain = RetrievalQA\
        .from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=index.vectorstore.as_retriever(), 
            input_key="question"
        )


while True:
    query = console.input("Ask DQA: (enter q for quit): ")
    if "q" == query.lower():
        console.print("[red blink]Exiting...!")
        break
    start = datetime.now()
    console.print("[red blink]Executing...")
    console.print(f"[grey78]Generating answer to your question:[grey78] [green_yellow]{query}")
    response = chain.run(query)
    wrapped_text = textwrap.fill(response, 100)
    console.print(Panel(wrapped_text, title="DQA Reply", title_align="center"))
    stop = datetime.now()
    elapsed = stop - start
    console.rule(f"Report Generated in {elapsed}")
    console.print(f"DQA @ {datetime.now().ctime()}")