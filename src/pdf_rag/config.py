from jinja2 import Environment, PackageLoader

jinja2_env = Environment(
    loader=PackageLoader(package_name="pdf_rag", package_path="templates"),
    trim_blocks=True,
    lstrip_blocks=True,
)