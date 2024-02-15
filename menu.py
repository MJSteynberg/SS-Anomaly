from nicegui import ui

def menu() -> None:
    """
    Creates a menu using the ui.link method from the nicegui library.
    Adds links to different pages with specific styles and classes.
    """
    links = [
        ('Homepage', '/'),
        ('PDF Summarizer', '/PDF_Summarizer'),
        ('Cleaning', '/cleaning'),
        ('VPIN', '/vpin'),
        ('Signature', '/signature')
    ]
    
    for link_text, link_url in links:
        ui.link(link_text, link_url).classes(replace='text-white').style('font-size: 16px;')