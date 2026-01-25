from flask import Flask, render_template, jsonify, request
from torch import embedding
from src.helpers import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embedding = download_embeddings()

index_name = "medical" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "Erreur : aucun message reçu", 400

    try:
        response = rag_chain.run(msg)
        return response
    except Exception as e:
        # Imprime l'erreur complète dans la console
        print("Erreur LLM / RAG:", str(e))
        return f"Erreur serveur : {str(e)}", 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)