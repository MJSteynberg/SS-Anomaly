from nicegui import ui, app, events, native
from nicegui.events import ValueChangeEventArguments
from Signatures.anomaly_backend import *
import home_page    
import theme
import Cleaning.cleaning_page as cleaning_page
import VPIN.vpin_page as vpin_page
import Signatures.signature_page as signature_page
import PDF.frontend as frontend

@ui.page('/')
def home():
    """
    Renders the content of the home page within a common page frame.
    """
    with theme.frame('Homepage'):
        home_page.content()

@ui.page('/PDF_Summarizer')
def PDF_Summarizer():
    """
    Renders the content of the PDF Summarizer page within a common page frame.
    """
    with theme.frame('PDF Summarizer'):
        frontend.content()

@ui.page('/cleaning')
async def cleaning():
    """
    Renders the content of the cleaning page within a common page frame.
    """
    with theme.frame('Cleaning'):
        cont = cleaning_page.Content()  
        await cont.content()

@ui.page('/vpin')
def vpin():
    """
    Renders the content of the VPIN page within a common page frame.
    """
    with theme.frame('VPIN'):
        vpin_page.content()

@ui.page('/signature')
def signature():
    """
    Renders the content of the signature page within a common page frame.
    """
    with theme.frame('Signature'):
        signature_page.content()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Signature Anomaly Detection Methods")