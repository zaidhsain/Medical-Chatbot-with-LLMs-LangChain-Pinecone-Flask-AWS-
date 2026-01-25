from flask import Flask
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Flask
app = Flask(__name__)

# Charger .env
load_dotenv()

print("OPENAI_KEY:", os.getenv("OPENAI_API_KEY"))
print("BASE_URL:", os.getenv("OPENAI_BASE_URL"))

# LLM OpenRouter minimal
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

@app.route("/")
def index():
    try:
        # Test simple
        response = llm.invoke("Say hello from Flask")
        return f"LLM Response: {response.content}"
    except Exception as e:
        return f"ERROR: {str(e)}"

