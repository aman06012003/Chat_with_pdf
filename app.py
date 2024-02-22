import os
# from transformers import pipeline
from langchain import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from typing import Any
from langchain.document_loaders import (
  PyPDFLoader, TextLoader,
  UnstructuredWordDocumentLoader,
  UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import pathlib
from langchain.schema import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.agents import (
    AgentExecutor, AgentType, initialize_agent, load_tools
)
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatFireworks
from langchain.memory import ConversationBufferMemory



os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ["FIREWORKS_API_KEY"] = "JYcjITeKhnB2ZKAM544JtTtqp7lwh0mOs7GllwmPCdySp2m8"


class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], ** kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")
class DocumentLoaderException(Exception):
    pass
class DocumentLoader(object):
    # """Loads in a document with a supported extension."""
    supported_extentions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }

    
def load_document(temp_filepath: str) -> list[Document]:
    # """Load a file and return it as a list of documents."""
        ext = pathlib.Path(temp_filepath).suffix
        loader = DocumentLoader.supported_extentions.get(ext)
        if not loader:
            raise DocumentLoaderException(
                f"Invalid extension type {ext}, cannot load this type of file"
            )
        loader = loader(temp_filepath)
        docs = loader.load()
        logging.info(docs)
        return docs

def configure_retriever(docs: list[Document]) -> BaseRetriever:
    # """Retriever to use."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})


def configure_chain(retriever: BaseRetriever) -> Chain:
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Setup LLM and QA chain; set temperature low to keep hallucinations in check
    llm = ChatFireworks(temperature=0.5, streaming=True)
    
    # Passing in a max_tokens_limit amount automatically
    return ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True, max_tokens_limit=4000)



def load_agent() -> AgentExecutor:
    llm = ChatFireworks(temperature=0, streaming=True)
    tools = load_tools(
        tool_names=["ddg-search","arxiv", "wikipedia"],
        llm=llm
    )
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

import os
import tempfile
def configure_qa_chain(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))
    retriever = configure_retriever(docs=docs)
    return configure_chain(retriever=retriever)


import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch

st.set_page_config(page_title="Chat with documents")
st.title(" LangChain: Chat with Documents")
uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=list(DocumentLoader.supported_extentions.keys()),
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()
qa_chain = configure_qa_chain(uploaded_files)
assistant = st.chat_message("assistant")
user_query = st.chat_input(placeholder="Ask me anything!")
if user_query:
    stream_handler = StreamlitCallbackHandler(assistant)
    response = qa_chain.run(user_query, callbacks=[stream_handler])
    st.markdown(response)
