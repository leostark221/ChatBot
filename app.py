from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Lazy loading components
embeddings = None
docsearch = None
retriever = None
rag_chain = None


def initialize_resources():
    global embeddings, docsearch, retriever, rag_chain

    if embeddings is None:
        print("Downloading embeddings...")
        embeddings = download_hugging_face_embeddings()

    if docsearch is None:
        print("Initializing Pinecone VectorStore...")
        index_name = "mentalhealth-bot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

    if retriever is None:
        print("Setting up retriever...")
        retriever = docsearch.as_retriever(
            search_type="similarity", search_kwargs={"k": 3})

    if rag_chain is None:
        print("Setting up RAG chain...")
        llm = OpenAI(temperature=0.4, max_tokens=500)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    global rag_chain

    # Initialize resources if not already initialized
    initialize_resources()

    msg = request.form["msg"]
    print(f"User input: {msg}")
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run()
