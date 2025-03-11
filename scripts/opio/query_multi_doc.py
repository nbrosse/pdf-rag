import os
from pathlib import Path
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

from pdf_rag.agents_multi_doc import ReActAgentMultiDoc
from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from google import genai

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

# Paths
opio_dir_path = Path("/home/nicolas/Documents/projets/opio/")
uses_cases_path = opio_dir_path / "raw data" / "use cases"
orano_path = uses_cases_path / "4 - donnees demo Orano RAA 2016-2023"
shell_path = uses_cases_path / "Shell Dec 5 2024"
cache_dir = opio_dir_path / "cache"
storage_dir = opio_dir_path / "storage"


react_agent_multi_doc = ReActAgentMultiDoc(
    root_dir=uses_cases_path,
    storage_dir=storage_dir,
    cache_dir=cache_dir,
    num_workers=16,
)

questions = [
    "Quel est le chiffre d'affaire d'Orano sur la période 2016 - 2023 ?",
    "Donne-moi les faits marquants sur le secteur de l'aval du cycle depuis 2016.",
    "Peux-tu me donner le tableau d'évolution de la provision pour obligation de fin de cycle depuis 2018 ?",
    "Peux-tu me donner le détail des dotations et des reprises sur la provision, ainsi que les soldes d'ouverture et de clôture en 2023 ?",
    "Donne moi le tableau de variation des immobilisations corporelles en 2023 ?",
    "Donne moi le détail des autres produits et charges opérationnels de 2023 ?",
    "Peux tu restituer le tableau des autres charges opérationnelles ?",
    "Quelle est l'évolution des difficultés de production à la Hague depuis 2020 ?",
    "Est-ce qu'il y a eu des difficultés spécifiques en 2020 ?",
    "Est-ce qu'il y a eu des difficultés opérationnelles particulières sur le site de la Hague en 2020 ?"
]


def main():
    responses = react_agent_multi_doc.run(
        dir_path=orano_path,
        queries=questions,
    )
    for q, r in zip(questions, responses):
        print(30 * "-")
        print(q)
        print(r)


if __name__ == "__main__":
    main()