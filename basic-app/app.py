import logging
from shiny.express import input, ui, render
from pathlib import Path
from llama_index.core.storage.docstore import SimpleDocumentStore
import json

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the docstore
data_dir = Path("/home/nicolas/Documents/projets/pdf-rag/data/storage/")
docstore = SimpleDocumentStore.from_persist_path(str(data_dir / "processed_docstore_storage.json"))

def format_metadata_value(value):
    
    # Try to parse as JSON if it looks like a dict/list string
    if isinstance(value, str):
        if (value.startswith('{') and value.endswith('}')) or \
           (value.startswith('[') and value.endswith(']')):
            try:
                parsed = json.loads(value)
                return ui.tags.pre(json.dumps(parsed, indent=2, ensure_ascii=False))
            except:
                pass
        
        # Handle multiline strings
        if '\n' in value:
            return ui.tags.pre(value)
    
    # Handle lists and dicts directly
    if isinstance(value, (dict, list)):
        return ui.tags.pre(json.dumps(value, indent=2, ensure_ascii=False))
    
    # Default case
    return str(value)

# Page configuration
ui.page_opts(title="Document Store Explorer", fillable=True)

# Summary section
with ui.card(min_height=200):
    ui.h2("Document Store Summary")
    @render.text
    def doc_count():
        return f"Total documents: {len(docstore.docs)}"

# Document selection and display
with ui.card():
    with ui.layout_sidebar():
        with ui.sidebar():
            ui.h3("Document Selection")
            # Create dropdown with document IDs instead of filenames
            @render.ui
            def doc_selector():
                choices = [(doc_id, doc.metadata.get("filename", doc_id)) for doc_id, doc in docstore.docs.items()]
                return ui.input_select(
                    "selected_doc_id",
                    "Select Document",
                    choices=dict(choices)
                )

        ui.h2("Document Details")
        
        @render.express
        def metadata_panel():
            if not input.selected_doc_id():
                return ui.p("Please select a document")
            
            try:
                # Direct document lookup by ID
                doc = docstore.docs.get(input.selected_doc_id())
                if not doc:
                    logger.error(f"Document not found: {input.selected_doc_id()}")
                    return ui.p("Error: Document not found")

                # Display metadata
                with ui.div():
                    ui.h3(f"Metadata for: {doc.metadata.get('filename', 'Unknown')}")
                    with ui.tags.table(class_="table table-striped"):
                        with ui.tags.thead():
                            with ui.tags.tr():
                                ui.tags.th("Property")
                                ui.tags.th("Value")
                        with ui.tags.tbody():
                            for key, value in doc.metadata.items():
                                with ui.tags.tr():
                                    ui.tags.td(key)
                                    ui.tags.td(format_metadata_value(value))
                    
                    ui.h3("Content Preview")
                    ui.markdown(doc.text[:500])
                    
            except Exception as e:
                logger.exception("Error displaying document metadata")
                return ui.p(f"Error: {str(e)}")