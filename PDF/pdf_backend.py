from typing import List
from openai import Client
import os
from openai import OpenAI
from PyPDF2 import PdfReader
import apikey
import io
import PDF.key as key
apikeyVal = key.apiKey()

client = OpenAI(api_key=apikeyVal)

def get_pdf_summary(files: List[str], log, system_role: str, user_role: str, model: str) -> List[str]:
    """
    This function takes in a list of files, a log, a system role, and a user role as inputs.
    It reads each PDF file in the list, extracts the text from the pages, and sends it as a message to the OpenAI chat completions API.
    The API generates a summary of the text and the function saves the summary to a new file with a "_summary.txt" suffix.
    Finally, it returns a list of the filenames of the summary files.
    """
    all_file_text = []
    for file in files:
        filename = file.name
        log.push(filename)
        pdf_summary_text = f"{filename}\n\n\n"
        
        # Read the PDF file using PyPDF2
        pdf_reader = PdfReader(io.BytesIO(file.content.read()))
        page_text = ' '.join([pdf_reader.pages[i].extract_text().lower() for i in range(len(pdf_reader.pages))])
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_role + page_text},
            ],
        )
        page_summary = response.choices[0].message.content

        pdf_summary_text += f"{page_summary}\n\n"
        filename = filename.replace(os.path.splitext(filename)[1], "_summary.txt")
        with open(filename, "w") as f:
            f.write(pdf_summary_text)
        all_file_text.append(filename)
    
    log.push("Done")
    return all_file_text
                