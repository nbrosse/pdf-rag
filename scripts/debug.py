import os

from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], verbose=True)

response = agent.chat("What is 2123 * 215123")