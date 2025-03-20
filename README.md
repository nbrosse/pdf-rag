# pdf-rag

PDF RAG is a tool designed to enhance PDF processing and querying using Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) techniques.  This repository accompanies the blog post [PDF RAG: Enhanced PDF Processing](https://nbrosse.github.io/posts/pdf-rag/pdf-rag.html). It provides two key functionalities:

1.  **PDF Structure Parsing**:  Leverages the Gemini LLM to extract the structural elements and metadata from PDF documents, providing a richer understanding of the document's organization.
2.  **Multi-PDF Querying**:  Employs a RAG model to query multiple PDFs simultaneously.

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:nbrosse/pdf-rag.git
   # or git clone https://github.com/nbrosse/pdf-rag.git
   cd pdf-rag
   ```

2.  **Install dependencies:**

    ```bash
    uv sync
    ```
    *Note:  Ensure you have [uv](https://astral.sh/uv) installed.*

## Usage

### 1. Process PDF Structure (Metadata and Structure Extraction)

This section outlines how to extract metadata and parse the structure of PDF documents using the `scripts/metadata_structure.py` script.

**Configuration**

1.  Copy the example configuration file:

    ```bash
    cp configs/metadata_structure_example.yaml configs/metadata_structure.yaml
    ```

2.  Edit the configuration file (`configs/metadata_structure.yaml`): Update the parameters to reflect your environment. Here's a breakdown of the configuration options:

    ```yaml
    data_dir: "/path/to/data"  # Path to the directory containing your PDF files.  Example: the path to the "data" folder.
    erase: false              # If true, the existing cache data will be cleared before processing.
    api_key_gemini: "your_gemini_api_key" # Your Google Gemini API key. Required.
    api_key_mistral: "your_mistral_api_key" # Your Mistral AI API key. Required.
    num_workers: 16            # Number of parallel async processes to use for PDF processing.
    neo4j_uri: "neo4j+s://xxxx.databases.neo4j.io"  # URI of your Neo4j database.  Only required if you want to store the document relationships in Neo4j.
    neo4j_username: "neo4j"      # Neo4j username.
    neo4j_password: "your_neo4j_password" # Neo4j password.
    neo4j_database: "neo4j"      # Neo4j database name.
    pipeline_step: "all"           # Select the pipeline stage to run:
                                    #  "index": Convert PDFs to Markdown, extract metadata/structure, and store in docstore.
                                    #  "tree": Load documents from docstore, build tree index, optionally export to Neo4j.
                                    #  "all": Run the complete pipeline (index and tree).
    ```

    *   **Important:**  `data_dir` must contain a folder named `pdfs` with your PDF files.
    *   Obtain API keys for Gemini and Mistral from their respective platforms.
    *   The Neo4j configuration is optional. If you don't need to export to Neo4j, you can leave these fields blank.

**Execution**

1.  Run the script:

    ```bash
    uv run scripts/metadata_structure.py --config configs/metadata_structure.yaml
    ```

**Output**

After successful execution, you'll find the following directory structure within your `data` folder (assuming you've used the default data directory structure):

```
data/
├── cache/           # Contains Markdown files converted from PDFs (potentially reformatted).
├── pdfs/            # Contains your original PDF files.
└── storage_metadata/ # Stores pipeline documents with metadata and structure in JSON format (processed_docstore_storage.json).
```

**Visualizing Metadata and Structure (Shiny App)**

A Shiny app is provided to visualize the extracted metadata and document structure.

1.  Modify the app's data path:  Edit `app/app-metadata.py` to ensure the path to the `processed_docstore_storage.json` file is correct:

    ```python
    # app/app-metadata.py
    from pathlib import Path
    # ...
    data_dir = Path(__file__).parents[1] / "data"
    docstore = SimpleDocumentStore.from_persist_path(str(data_dir / "storage_metadata" / "processed_docstore_storage.json"))
    # ...
    ```

2.  Run the Shiny app:

    ```bash
    uv run shiny run --reload --launch-browser app/app-metadata.py
    ```

    This will launch the Shiny app in your web browser.

**Visualizing with Neo4j (Optional)**

If you configured Neo4j, you can visualize the document structure and relationships in a graph database. Refer to the blog post [PDF RAG: Enhanced PDF Processing](https://nbrosse.github.io/posts/pdf-rag/pdf-rag.html) for examples.

### 2. Query Multiple PDFs (RAG-based Querying)

This section explains how to query multiple PDF documents using a RAG model via the `scripts/query_multi_pdfs.py` script.

**Configuration**

1.  Copy the example configuration file:

    ```bash
    cp configs/query_multi_pdfs_example.yaml configs/query_multi_pdfs.yaml
    ```

2.  Edit the configuration file (`configs/query_multi_pdfs.yaml`): Update the parameters according to your needs.

    ```yaml
    api_key_gemini: "your_gemini_api_key" # Your Google Gemini API key. Required.
    api_key_mistral: "your_mistral_api_key" # Your Mistral AI API key. Required.
    data_dir: "/path/to/data"         # Path to the directory containing the data (PDFs, cache, metadata).  Example: the path to the "data" folder.
    num_workers: 16                  # Number of workers for processing.
    chunks_top_k: 5                  # Number of top chunks to retrieve per document during RAG.
    nodes_top_k: 10                  # Number of top documents to retrieve based on initial query.
    max_iterations: 20               # Maximum iterations for the ReAct agent.
    verbose: true                    # Print progress and intermediate steps during querying.
    queries:                         # List of queries to execute.
      - "What are the vulnerabilities introduced by relying on application programming interfaces (APIs) in Banking as a Service (BaaS)?"
      - "What mitigation opportunities are there to ensure strong security for BaaS platforms and API connectivity?"
      - "How can the industry best improve due diligence on BaaS providers in this landscape?"
      - "What are the common objectives of the Open Data ecosystem?"
      - "What are key strategic decisions to be made by ecosystem participants?"
      - "How can the public and private sectors collaborate to promote innovation, secure data sharing, and data privacy within the Open Data ecosystem?"
      - "What are the key characteristics that define the Leaders, Major Contenders, and Aspirants within the Life Sciences Smart Manufacturing Services PEAK Matrix?"
      - "What are some of the solutions that can assist biopharma and MedTech manufacturers with insights that help optimize manufacturing processes and improve product quality?"
      - "How are different areas of a manufacturing line benefiting from AI?"
      - "What are ConocoPhillips' key financial priorities for the next 10 years?"
      - "How does ConocoPhillips plan to meet it's Net-Zero and Emissions targets?"
      - "What is ConocoPhillips strategy to grow production?"
      - "What are the key features and capabilities of the XC9500 In-System Programmable CPLD family?"
      - "How does the Fast CONNECT switch matrix enable flexible signal routing and logic implementation within the XC9500 devices?"
      - "What design security options are available in the XC9500 family, and how do they protect programming data?"
      - "What strategies are companies implementing to address tax transformation in a data-driven world?"
      - "How can tax departments effectively invest in technology to meet compliance and strategic goals?"
      - "How can tax departments collaborate internally to secure budget and ensure technology alignment?"
    ```

    *   **Important**: `data_dir` must contain a folder named `pdfs` with your PDF files.
    *   Set `api_key_gemini` and `api_key_mistral` to your valid Gemini and Mistral API keys.
    *   Adjust `chunks_top_k` and `nodes_top_k` to control the granularity and scope of the RAG process. Higher values may improve accuracy but also increase processing time.
    *   Modify the `queries` list to reflect the questions you want to ask of your PDF documents.

**Execution**

1.  Run the script:

    ```bash
    uv run scripts/query_multi_pdfs.py --config configs/query_multi_pdfs.yaml
    ```

    The script will process the queries and print queries and responses to the console.

**Interactive Querying (Notebook)**

For interactive querying, use the provided Jupyter notebook: `notebooks/query_multi_pdfs_example.ipynb`. Open the notebook in Jupyter and execute the cells to run the queries and explore the results.

**Output**

After successful execution, you'll find the following directory structure within your `data` folder (assuming you've used the default data directory structure):

```
data/
├── cache/           # Contains Markdown files converted from PDFs (potentially reformatted).
├── pdfs/            # Contains your original PDF files.
└── storage_queries/  # Stores document indices for efficient querying and retrieval.
```

## Project Structure

```
pdf-rag/
├── README.md
├── LICENSE
├── pyproject.toml
├── .python-version
├── app/
│   └── app-metadata.py
├── configs/
│   ├── metadata_structure_example.yaml
│   └── query_multi_pdfs_example.yaml
├── data/
│   └── pdfs/
├── notebooks/
│   └── query_multi_pdfs_example.ipynb
├── scripts/
│   ├── metadata_structure.py
│   └── query_multi_pdfs.py
└── src/
    └── pdf_rag/
        ├── __init__.py
        ├── config.py
        ├── extractors.py
        ├── gemini_wrappers.py
        ├── markdown_parsers.py
        ├── node_postprocessors.py
        ├── react_agent_multi_pdfs.py
        ├── readers.py
        ├── structure_parsers.py
        ├── transforms.py
        ├── tree_index.py
        ├── utils.py
        └── templates/
```

## Source PDFs

For the sources of test PDF files, refer to the [pdf-parsing repository](https://github.com/nbrosse/pdf-parsing#sources-of-pdf).