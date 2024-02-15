import nicegui.ui as ui
import Cleaning.cleaning_backend as cb
import asyncio
from nicegui import app
from os import listdir
import os

class Content:
    """
    The `Content` class is responsible for managing the UI components and handling user interactions.
    """

    def __init__(self):
        """
        Initializes the `Content` class and sets the initial values for the time and volume bucketing flags,
        as well as the bucket sizes and log component.
        """
        self.log = None

            
    def clicked(self):
        """
        This function is called when the user clicks the "Run Algorithm" button.
        It calls the `handle_cleaning` function from the `cleaning_backend` module.
        """
        cb.button_callback(self.trim.value, self.K.value, self.Gamma.value, self.price_cleaning.value, self.volume_cleaning.value, self.start.value, self.end.value, self.running_sum_n.value, self.skip_lines.value, self.time_bucket.value, self.time_bucket_size.value, self.volume_bucket.value, self.volume_bucket_size.value, self.log)

    async def start_route(self):
        """Start the route for the log component."""
        asyncio.run(await cb.start_stream(self.log))

    def clear(self):
        # Delete all files with .pdf
        all_files = listdir()
        
        for file in all_files:
            if file.endswith(".xlsx"):
                os.remove(file)
        try:
            self.log.clear()
        except:
            pass

    def content(self) -> None:
        """
        This function creates the UI content and handles user interactions for the cleaning algorithm.
        """
        ui.upload(on_upload=cb.handle_upload, label="Upload input file").classes('max-w-full')
        with ui.row():
            with ui.column().classes('#dbeafe items-stretch p-4'):
                self.trim = ui.number('Trim value', value=0.01)
                self.K = ui.number('K value', value=60)
                self.Gamma = ui.number('Gamma value', value=0.02)
                with ui.row():
                    self.price_cleaning = ui.checkbox('Price based cleaning', value=False)
                    self.volume_cleaning = ui.checkbox('Volume based cleaning', value=True)
                self.start = ui.input('Start time', value='09:00')
                self.end = ui.input('End time', value='17:00')
                self.running_sum_n = ui.input('Volatility running sum n', value='10')
                self.skip_lines = ui.number('Skip the first lines', value=0)
                self.time_bucket = ui.checkbox('Time bucket')
                self.time_bucket_size = ui.number('Time bucket size (minutes)', value=10)
                self.volume_bucket = ui.checkbox('Volume bucket')
                self.volume_bucket_size = ui.number('Volume bucket size (trades)', value=10)
            
            with ui.column().classes('#dbeafe items-stretch p-4'):
                self.log = ui.log(20).classes("w-full").style("height: 400px; width: 350px; color: #ffffff; background-color: #000000; ")
                app.on_connect(self.start_route)
                ui.space()
                ui.button('Run Algorithm').on('click', self.clicked).classes(
                    'bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full')
                ui.button("Download all pdfs", on_click=cb.download_pdf).classes(
                    'bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full')
                ui.button("Clear All", on_click=self.clear).classes(
                    'bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full')