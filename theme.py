from typing import Iterator
from contextlib import contextmanager

from menu import menu

from nicegui import ui


@contextmanager
def frame(navtitle: str) -> Iterator[None]:
    """
    Custom page frame to share the same styling and behavior across all pages.

    Args:
        navtitle: The title to be displayed in the navigation menu.

    Yields:
        None

    """
    # Set the color scheme for the UI
    ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E', positive='#53B689')

    # Create the header with title and navigation menu
    with ui.header().classes('justify-between text-white'):
        ui.label('S-Software Design Market Anomaly Detection').style('font-size: 16px; font-weight: bold;')
        ui.label(navtitle).classes('font-bold text-2xl absolute-center items-center')
        with ui.row():
            menu()
    
    # Create a column to center the content of the page
    with ui.column().classes('absolute-center items-center'):
        ui.space().style('height: 400px;')
        yield