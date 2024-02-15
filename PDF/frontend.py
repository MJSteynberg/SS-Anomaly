from nicegui import ui, native, app
import PDF.backend as b
import asyncio
from nicegui import ui, app, events
import time
import asyncio
import sys
from io import StringIO
import io
import os
from logging import getLogger, StreamHandler
from PyPDF2 import PdfReader

async def button_click(folder, log, system_role, user_role):
    global all_file_text
    all_file_text = await asyncio.to_thread(b.get_pdf_summary, folder, log, system_role, user_role)


def download_pdf(log):
    global all_file_text
    log.push("Downloading PDFs")
    print(all_file_text)
    for text in all_file_text:
         ui.download(text)


def clear(log):
    global all_file_text
    for text in all_file_text:
        os.remove(text)
    all_file_text = []
    log.clear()
    upload_func.refresh()

    
@ui.refreshable
def upload_func():
    global file_list
    file_list = []
    upload = ui.upload(
        label="Upload all PDF files to summarize:",
        on_upload=file_list.append,
        auto_upload=True,
        multiple=True,
    )
    return upload
def content():
    with ui.row():
        with ui.column():
            upload_func()
            system_role = ui.textarea("System Role", value="Give the title and authors of the paper and summarise the main sections: ")
            user_role = ui.textarea("User Role", value="You are a helpful research assistant.")

        with ui.column():
            log = ui.log(20).classes("w-full").style("height: 400px; width: 350px; color: #ffffff; background-color: #000000; ")
            ui.button("Summarise all pdfs", on_click=lambda: button_click(file_list, log, system_role, user_role))
            ui.button("Download all pdfs", on_click=lambda: download_pdf(log))
            ui.button("Clear All", on_click=lambda: clear(log))
    ui.space().style('height: 100px;')


