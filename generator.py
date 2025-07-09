# generator.py

import os
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult
from langchain_community.llms import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL_NAME = os.getenv("TOGETHER_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

if not TOGETHER_API_KEY:
    raise ValueError("❌ TOGETHER_API_KEY not found in .env file.")

# Initialize TogetherAI LLM
llm: BaseLLM = Together(
    model=TOGETHER_MODEL_NAME,
    temperature=0.3,
    max_tokens=512,
    top_p=0.95,
    together_api_key=TOGETHER_API_KEY,
)

def generate_answer(prompt: str) -> str:
    """
    Generate a response using TogetherAI LLM.
    
    Args:
        prompt (str): The input prompt/question.

    Returns:
        str: The generated response.
    """
    try:
        response: str = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"❌ Error in LLM generation: {e}")
        return "Sorry, something went wrong while generating the answer."

# Optional CLI interface for quick testing
if __name__ == "__main__":
    while True:
        user_input = input("🧠 Ask AI-LAWS: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(user_input)
        print(f"\n💬 {answer}\n")
