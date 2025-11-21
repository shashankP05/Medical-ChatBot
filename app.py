from flask import Flask, render_template, jsonify, request
from src.gemini_runnable import GeminiRunnable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# Only set environment variables if values are present to avoid setting None
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = GeminiRunnable(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = prompt | llm
def rag_retriever(input):
    return retriever.invoke(input["input"])

rag_chian = (
    {"context": rag_retriever, "input": RunnablePassthrough()}
    | question_answer_chain
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    # Pass a predictable dict so the Grok wrapper can extract the prompt consistently
    response = rag_chian.invoke({"input": msg})
    print("response: ", response["content"])
    return str(response["content"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
    