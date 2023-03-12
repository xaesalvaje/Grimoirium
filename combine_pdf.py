import os
from PyPDF2 import PdfReader, PdfMerger

def merge_pdfs(input_dir, output_path):
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    pdf_files.sort()
    pdf_merger = PdfMerger()

    for pdf_file in pdf_files:
        pdf_merger.append(PdfReader(os.path.join(input_dir, pdf_file), "rb"))

    with open(output_path, "wb") as outfile:
        pdf_merger.write(outfile)

    print(f"Successfully merged PDFs into {output_path}")

# Set the paths to the input and output folders
input_dir = "./data/input"
output_dir = "./data/output"

# Merge the PDF files into a single file
output_path = os.path.join(output_dir, "merged.pdf")
merge_pdfs(input_dir, output_path)

print("PDF files merged successfully!")
