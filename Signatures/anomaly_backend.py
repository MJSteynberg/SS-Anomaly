from nicegui import events, ui, app
from nicegui.events import ValueChangeEventArguments
import pandas as pd 
import Signatures.Path as p
import Signatures.RandSig as rs
import io
from io import StringIO
import plotly.graph_objects as go
import asyncio
import sys
from logging import getLogger, StreamHandler

def show(event: ValueChangeEventArguments):
    """
    Display a notification with the name of the sender and the new value.

    Args:
        event (ValueChangeEventArguments): The event object that contains information about the value change event.

    Returns:
        None
    """
    name = type(event.sender).__name__
    value = event.value
    notification = f'{name}: {value}'
    ui.notify(notification)
with ui.header():
    ui.label('Signature Anomaly Detection Methods').classes('text-3xl font-bold')


logger = getLogger(__name__)
logger.setLevel("DEBUG")

async def button_callback(volume: bool, price: bool, signed_volume: bool, datalength: int, price_plot: bool, volume_plot: bool, signed_volume_plot: bool, path_transform: str, clustering: str, sensitivity: float, reservoir_dim: int, input_dim: int, fig, plot):
    """Called after button click"""
    await asyncio.to_thread(perform_clustering, volume, price, signed_volume, datalength, price_plot, volume_plot, signed_volume_plot, path_transform, clustering, sensitivity, reservoir_dim, input_dim, fig, plot)

def clear(fig, plot, log):
    """
    Clears the data from a plot and the log component in a GUI application.

    Args:
        fig (plot figure object): The plot figure object that needs to be cleared.
        plot (plot component object): The plot component object that needs to be updated.
        log (log component object): The log component object that needs to be cleared.

    Returns:
        None
    """
    fig.data = []
    plot.update()
    log.clear()

async def start_stream(log):
    """Start a 'stream' of console outputs."""
    # Create buffer
    string_io = StringIO()
    # Standard ouput like a print
    sys.stdout = string_io
    # Errors/Exceptions
    sys.stderr = string_io
    # Logmessages
    stream_handler = StreamHandler(string_io)
    stream_handler.setLevel("DEBUG")
    logger.addHandler(stream_handler)
    while 1:
        await asyncio.sleep(2)  # need to update ui
        # Push the log component and reset the buffer
        log.push(string_io.getvalue())
        string_io.truncate(0)

def handle_upload(e: events.UploadEventArguments):
    """
    Process an uploaded file by reading its content, selecting specific columns, and storing the resulting DataFrame in a global variable.

    Args:
        e (events.UploadEventArguments): An object that contains information about the uploaded file.

    Returns:
        None
    """
    # Display a notification indicating that a file has been uploaded
    ui.notify(f'Uploaded {e.name}')
    try:
        # Read the content of the uploaded file as a DataFrame
        df = pd.read_excel(io.BytesIO(e.content.read()))

        # Select specific columns from the DataFrame
        selected_columns = ['Date_Time', 'Volume', 'Price', 'Signed_Volume', 'Trading_Time_Indicator']
        df = df[selected_columns]

        # Filter the DataFrame based on a condition
        filtered_df = df[df['Trading_Time_Indicator'] == 'In']

        # Store the filtered DataFrame in a global variable
        global global_data
        global_data = filtered_df
    except:
        raise Exception('No file uploaded: Please upload a correct input file')
        return

def perform_clustering(volume: bool, price: bool, signed_volume: bool, datalength: int, price_plot: bool, volume_plot: bool, signed_volume_plot: bool, path_transform: str, clustering: str, sensitivity: float, reservoir_dim: int, input_dim: int, fig, plot):
    """
    Perform clustering on input data based on selected features and clustering algorithm, and plot the results using Plotly.

    Args:
        volume (bool): Whether to include the volume feature in the clustering process.
        price (bool): Whether to include the price feature in the clustering process.
        signed_volume (bool): Whether to include the signed volume feature in the clustering process.
        datalength (int): The percentage of data to use for clustering.
        price_plot (bool): Whether to plot the price feature.
        volume_plot (bool): Whether to plot the volume feature.
        signed_volume_plot (bool): Whether to plot the signed volume feature.
        path_transform (str): The type of path transform to apply to the data.
        clustering (str): The clustering algorithm to use.
        sensitivity (float): The contamination/sensitivity parameter for the clustering algorithm.
        reservoir_dim (int): The reservoir dimension for the random signature.
        input_dim (int): The input dimension for the random signature.
        fig: The Plotly figure object to plot the results on.
        plot: The Plotly plot object to update after plotting the results.

    Returns:
        None
    """
    # Load the selected file and extract the selected features
    parameters = []
    if volume:
        parameters.append('Volume')
    if price:
        parameters.append('Price')
    if signed_volume:
        parameters.append('Signed_Volume')

    try:
        data = global_data[parameters].to_numpy()[:int(datalength * global_data.shape[0] / 100)]
    except:
        raise Exception('No file uploaded: Please upload a correct input file')
        return

    t = global_data['Date_Time'].to_numpy()[:int(datalength * global_data.shape[0] / 100)]
    t = pd.to_datetime(t)

    # Apply the path transform
    if path_transform == 'Lead Lag':
        data = p.lead_lag_transform(data)
    elif path_transform == 'Interpolation':
        data = p.interpolation(data)

    # Apply the random signature
    if clustering == 'Isolation Forest':
        random_sig = rs.RandomSig(reservoir_dim=reservoir_dim, input_dim=input_dim, anomaly=rs.IForest(contamination=sensitivity))
    elif clustering == 'LOF':
        random_sig = rs.RandomSig(reservoir_dim=reservoir_dim, input_dim=input_dim, anomaly=rs.LOF(contamination=sensitivity))

    labels = random_sig.fit(data)

    df = pd.DataFrame(labels, columns=['Labels_numeric'])
    df['Labels'] = df['Labels_numeric'].apply(lambda x: 'seagreen' if x == -1 else 'royalblue')

    # Plot the results
    print('Creating Plot')
    if price_plot:
        if input_dim == 1:
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 0], mode='markers', marker_color=df['Labels']))
        elif input_dim == 2 and volume and path_transform == 'Lead Lag':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 3], mode='markers', marker_color=df['Labels']))
        elif input_dim == 2 and volume and path_transform == 'Interpolation':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 1], mode='markers', marker_color=df['Labels']))
        else:
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 0], mode='markers', marker_color=df['Labels']))
    if volume_plot:
        fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 0], mode='markers', marker_color=df['Labels']))
    if signed_volume_plot:
        if input_dim == 1:
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 0], mode='markers', marker_color=df['Labels']))
        elif input_dim == 2 and path_transform == 'Lead Lag':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 3], mode='markers', marker_color=df['Labels']))
        elif input_dim == 2 and path_transform == 'Interpolation':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:, 1], mode='markers', marker_color=df['Labels']))

    plot.update()
