import os
from openai import OpenAI
from PyPDF2 import PdfReader
import apikey
import io
import PDF.key as key
apikeyVal = key.apiKey()

client = OpenAI(api_key=apikeyVal)

def get_pdf_summary(files, log, system_role, user_role):
    # Print number of files in the folder
    print(files)
    all_file_text = []
    for file in files:
        filename = file.name
        log.push(filename)
        pdf_summary_text = filename + "\n\n\n"
        # Read the PDF file using PyPDF2
        pdf_reader = PdfReader(io.BytesIO(file.content.read()))
        page_text = pdf_reader.pages[0].extract_text().lower() + ' '
        for i in range(len(pdf_reader.pages)):
            page_text += pdf_reader.pages[i].extract_text().lower()
            page_text += ' '
        
        response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {"role": "system", "content": system_role.value},
                            {"role": "user", "content": user_role.value + page_text},
                                ],
                                    )
        page_summary = response.choices[0].message.content

        pdf_summary_text+=page_summary  + "\n\n"
        filename = filename.replace(os.path.splitext(filename)[1], "_summary.txt")
        open(filename, "w").write(pdf_summary_text)
        all_file_text.append(filename)
    log.push("Done")
    return all_file_text
                