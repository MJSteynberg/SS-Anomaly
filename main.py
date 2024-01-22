from nicegui import ui, app, events
from nicegui.events import ValueChangeEventArguments
from backend import *

def __main__():
    """
    Main function for running the anomaly detection algorithm.

    This function sets up the user interface and handles user interactions.
    It creates a plot using Plotly and allows the user to choose various parameters for the algorithm.
    The chosen parameters are then passed to the button_callback function when the "Run Algorithm" button is clicked.

    Parameters:
        None

    Returns:
        None
    """
    async def start_route():
        """Start the route for the log component."""
        asyncio.run(await start_stream(log))

    # Frontend
    
    ui.upload(on_upload=handle_upload, label="Upload input file").classes('max-w-full')
    
    with ui.splitter() as splitter:
        with splitter.before:    
            fig = go.Figure(layout=go.Layout(
                title=go.layout.Title(text="Anomaly Detection"),
                xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Time")),
                yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Volume/Price/Signed Volume")),
                width=1000,
                height=700,
                autosize=True
            ))
            fig.update_layout(margin=dict(l=30, r=20, t=50, b=20))
            plot = ui.plotly(fig).classes('#dbeafe items-stretch')
        with splitter.after:
            
            with ui.column().classes('#dbeafe items-stretch p-4'):

                with ui.row():
                    ui.label('Choose the input dimension:').style('align-items: left;')
                    ui.space()

                    input_dim = ui.toggle([1, 2], value=1)
                with ui.row():
                    ui.label('Choose the reservoir dimension: ').style('align-items: left;')
                    ui.space()

                    reservoir_dim = ui.toggle([100, 200, 300, 400, 500], value=100)
                with ui.row():
                    ui.label('Choose the length of data that must be used: (%)').style('align-items: left;')
                    ui.space()

                    datalength = ui.toggle([20, 40, 60, 80, 100], value=100)
                with ui.row():
                    ui.label('Choose the parameters:').style('align-items: left;')
                    ui.space()
                    price = ui.checkbox('Price', value=False)
                    volume = ui.checkbox('Volume', value=True)
                    signed_volume = ui.checkbox('Signed Volume', value=False)

                with ui.row():
                    ui.label('Which parameter must be plotted:').style('align-items: left;')
                    ui.space()

                    price_plot = ui.checkbox('Price', value=False)
                    volume_plot = ui.checkbox('Volume', value=True)
                    signed_volume_plot = ui.checkbox('Signed Volume', value=False)
                with ui.row():
                    ui.label('Choose the path transform:').style('align-items: left;')
                    ui.space()

                    path_transform = ui.toggle(['Lead Lag', 'Interpolation'], value='Lead Lag')
                with ui.row():
                    ui.label('Choose the clustering algorithm:').style('align-items: left;')
                    ui.space()

                    clustering = ui.toggle(['Isolation Forest', 'LOF'], value='Isolation Forest')
                with ui.row():
                    ui.label('Choose clustering sensitivity:').style('align-items: left;')
                    ui.space()

                    sensitivity = ui.toggle([0.001, 0.02, 0.05, 0.1, 0.4, 'auto'], value='auto')
                ui.space()
                ui.space()
                with ui.row():
                    ui.button('Clear plot').on('click', clear, ['fig', 'plot', 'log']).classes(
                        'bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full')
                    ui.space()
                    ui.button('Run Algorithm').on('click', lambda volume = volume, price = price, signed_volume = signed_volume, datalength = datalength, price_plot = price_plot, volume_plot = volume_plot, signed_volume_plot = signed_volume_plot, path_transform = path_transform, clustering = clustering, sensitivity = sensitivity, reservoir_dim = reservoir_dim, input_dim = input_dim, fig = fig, plot = plot: button_callback(volume, price, signed_volume, datalength, price_plot, volume_plot, signed_volume_plot, path_transform, clustering, sensitivity, reservoir_dim, input_dim, fig, plot)).classes(
                        'bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full')

                    log = ui.log(20).classes("w-full").style("height: 200px; color: #ffffff; background-color: #000000; ")
                    app.on_startup(start_route)

        with splitter.separator:
            with ui.column():
                ui.label('||').style('font-size: 30px; font-weight: bold;')


    ui.run(reload=False, native=True)

if __name__ == '__main__':
    __main__()