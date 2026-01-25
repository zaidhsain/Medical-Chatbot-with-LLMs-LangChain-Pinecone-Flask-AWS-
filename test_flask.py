from flask import Flask
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Flask
app = Flask(__name__)

# Charger .env
load_dotenv()

# Vérification
print("OPENAI_KEY:", os.getenv("OPENAI_API_KEY"))
print("BASE_URL:", os.getenv("OPENAI_BASE_URL"))

# LLM minimal OpenRouter
llm = ChatOpenAI(
    model_name="gpt-4o-mini",   # modèle OpenRouter valide
    temperature=0.3,
    max_tokens=200,
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@app.route("/")
def index():
    try:
        # Test direct
        response = llm([HumanMessage(content="Say hello from Flask")])
        return f"LLM Response: {response.content}"
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=8080)
