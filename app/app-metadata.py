import base64

from llama_index.core.schema import BaseNode
from shiny import ui, render, App
import logging
from pathlib import Path
from llama_index.core.storage.docstore import SimpleDocumentStore
import json

from pdf_rag.structure_parsers import parse_structure

root_logger = logging.getLogger()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
root_logger.setLevel(logging.ERROR)


parsed_structure_info = {
    "landscape": """
    Structure parsed from PDF using Gemini (landscape mode):
    - Indentation shows parent/child relationships
    - Numbers in [] represent:
      * Positive numbers: page numbers
      * Negative numbers: abstract nodes grouping related sections
    """,
    "portrait": """
    Structure parsed from PDF using Gemini (portrait mode):
    - Indentation shows parent/child relationships
    - Numbers in [] represent:
      * Positive numbers: line numbers
      * Negative numbers: abstract nodes grouping related sections
    """,
}

# Load the docstore
data_dir = Path(__file__).parents[1] / "data"
docstore = SimpleDocumentStore.from_persist_path(str(data_dir / "storage_metadata" / "processed_docstore_storage.json"))


def get_pdf_data_url(file_path):
    """Convert PDF file to data URL for embedding"""
    try:
        with open(file_path, "rb") as f:
            pdf_data = f.read()
            b64_data = base64.b64encode(pdf_data).decode()
            return f"data:application/pdf;base64,{b64_data}"
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return None


def get_str_structure(doc: BaseNode) -> str:
    """Return a string representation of the document structure."""
    tree = parse_structure(doc)
    return str(tree)


def format_metadata_value(value):
    # Try to parse as JSON if it looks like a dict/list string
    if isinstance(value, str):
        if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
            try:
                parsed = json.loads(value)
                return ui.tags.pre(json.dumps(parsed, indent=2, ensure_ascii=False))
            except:
                pass

        # Handle multiline strings
        if "\n" in value:
            return ui.tags.pre(value)

    # Handle lists and dicts directly
    if isinstance(value, (dict, list)):
        return ui.tags.pre(json.dumps(value, indent=2, ensure_ascii=False))

    # Default case
    return str(value)


# Define UI
def get_doc_choices():
    return [(doc_id, doc.metadata.get("filename", doc_id)) for doc_id, doc in docstore.docs.items()]


app_ui = ui.page_fluid(
    # Summary section
    ui.card(
        ui.h2("Document Store Summary"),
        ui.output_text("doc_count"),
        # min_height=200,
    ),
    # Document selection and display
    ui.card(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Document Selection"),
                ui.input_select("selected_doc_id", "Select Document", choices=dict(get_doc_choices())),
            ),
            ui.h2("Document display"),
            ui.output_ui("display_panel"),
            ui.h2("Document Details"),
            ui.output_ui("metadata_panel"),
        )
    ),
)


def server(input, output, session):
    @output
    @render.text
    def doc_count():
        return f"Total documents: {len(docstore.docs)}"

    @output
    @render.ui
    def display_panel():
        if not input.selected_doc_id():
            return ui.p("Please select a document")

        try:
            doc = docstore.docs.get(input.selected_doc_id())
            if not doc:
                logger.error(f"Document not found: {input.selected_doc_id()}")
                return ui.p("Error: Document not found")

            pdf_path = doc.metadata.get("file_path")
            pdf_data_url = get_pdf_data_url(pdf_path) if pdf_path else None

            return ui.div(
                ui.h3(f"Display panel for: {doc.metadata.get('filename', 'Unknown')}"),
                ui.row(
                    ui.column(
                        6,  # Left column (PDF)
                        ui.h4("PDF View"),
                        ui.tags.iframe(
                            src=pdf_data_url,
                            style="width: 100%; height: 800px; border: 1px solid #ddd;",
                            type="application/pdf",
                        )
                        if pdf_data_url
                        else ui.p("No PDF available"),
                    ),
                    ui.column(
                        6,  # Right column (Markdown)
                        ui.h4("Content"),
                        ui.div(
                            ui.markdown(doc.text),
                            style="height: 800px; overflow-y: auto; border: 1px solid #ddd; padding: 1rem;",
                        ),
                    ),
                ),
            )

        except Exception as e:
            logger.exception("Error displaying document metadata")
            return ui.p(f"Error: {str(e)}")

    @output
    @render.ui
    def metadata_panel():
        if not input.selected_doc_id():
            return ui.p("Please select a document")

        try:
            doc = docstore.docs.get(input.selected_doc_id())
            if not doc:
                logger.error(f"Document not found: {input.selected_doc_id()}")
                return ui.p("Error: Document not found")

            return ui.div(
                ui.h3(f"Metadata for: {doc.metadata.get('filename', 'Unknown')}"),
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(ui.tags.th("Property"), ui.tags.th("Value"))),
                    ui.tags.tbody(
                        *[
                            ui.tags.tr(ui.tags.td(key), ui.tags.td(format_metadata_value(value)))
                            for key, value in doc.metadata.items()
                        ]
                    ),
                    class_="table table-striped",
                ),
                ui.h3(f"Parsed Structure for: {doc.metadata.get('filename', 'Unknown')}"),
                ui.markdown(parsed_structure_info.get(doc.metadata.get("format", ""), "")),
                ui.tags.pre(get_str_structure(doc)),
            )

        except Exception as e:
            logger.exception("Error displaying document metadata")
            return ui.p(f"Error: {str(e)}")


app = App(app_ui, server)
