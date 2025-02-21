import base64
import os
from io import BytesIO
from pathlib import Path

from mistralai import Mistral

from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from pypdf import PdfReader, PdfWriter

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
# model = "mistral-large-latest"
model = "pixtral-large-latest"

prompt = """
You are a specialized document transcription assistant converting PDF documents to Markdown format.
Your primary goal is to create an accurate, complete, and well-structured Markdown representation.

<instructions>
1. Language and Content:
   - MAINTAIN the original document language throughout ALL content
   - ALL elements (headings, tables, descriptions) must use source language
   - Preserve language-specific formatting and punctuation
   - Do NOT translate any content

2. Text Content:
   - Convert all text to proper Markdown syntax
   - Use appropriate heading levels (# ## ###)
   - Preserve emphasis (bold, italic, underline)
   - Convert bullet points to Markdown lists (-, *, +)
   - Maintain original document structure and hierarchy

3. Visual Elements (CRITICAL):
   a. Tables:
      - MUST represent ALL data cells accurately in original language
      - Use proper Markdown table syntax |---|
      - Include header rows
      - Add caption above table: [Table X: Description] in document language
      
   b. Charts/Graphs:
      - Create detailed tabular representation of ALL data points
      - Include X/Y axis labels and units in original language
      - List ALL data series names as written
      - Add caption: [Graph X: Description] in document language
      
   c. Images/Figures:
      - Format as: ![Figure X: Detailed description](image_reference)
      - Describe key visual elements in original language
      - Include measurements/scales if present
      - Note any text or labels within images

4. Quality Requirements:
   - NO content may be omitted
   - Verify all numerical values are preserved
   - Double-check table column/row counts match original
   - Ensure all labels and legends are included
   - Maintain document language consistently throughout

5. Structure Check:
   - Begin each section with clear heading
   - Use consistent list formatting
   - Add blank lines between elements
   - Preserve original content order
   - Verify language consistency across sections
</instructions>
"""


client = Mistral(api_key=api_key)

opio_dir_path = Path("/home/nicolas/Documents/projets/opio/")
shell_dir_path = opio_dir_path / "raw data" / "Shell Dec 5 2024"
shell_2023_report_path = shell_dir_path / "shell-annual-report-2023.pdf"
shell_2022_report_path = shell_dir_path / "shell-annual-report-2022.pdf"

reader = PdfReader(str(shell_2023_report_path))

page = reader.pages[1]
writer = PdfWriter()
writer.add_page(page)
with BytesIO() as bytes_stream:
    writer.write(bytes_stream)
    bytes_stream.seek(0)
    page_content = bytes_stream.read()

# Convert PDF to image
images = convert_from_bytes(page_content)
image = images[0]  # Assuming you want the first page

# Save the image to a bytes buffer
with BytesIO() as image_bytes_stream:
    image.save(image_bytes_stream, format="PNG")
    image_bytes_stream.seek(0)
    image_data = image_bytes_stream.read()

# Encode the image data
image_data_base64 = base64.b64encode(image_data).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data_base64}",
            },
            {
                "type": "text",
                "text": prompt,
            },

        ]
    }
]
#
#
chat_response = client.chat.complete(
    model= model,
    messages = messages,
)
print(chat_response.choices[0].message.content)