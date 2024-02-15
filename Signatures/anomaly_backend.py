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
    name = type(event.sender).__name__
    ui.notify(f'{name}: {event.value}')
with ui.header():
    ui.label('Signature Anomaly Detection Methods').classes('text-3xl font-bold')


logger = getLogger(__name__)
logger.setLevel("DEBUG")

async def button_callback(volume: bool, price: bool, signed_volume: bool, datalength: int, price_plot: bool, volume_plot: bool, signed_volume_plot: bool, path_transform: str, clustering: str, sensitivity: float, reservoir_dim: int, input_dim: int, fig, plot):
    """Called after button click"""
    await asyncio.to_thread(perform_clustering, volume, price, signed_volume, datalength, price_plot, volume_plot, signed_volume_plot, path_transform, clustering, sensitivity, reservoir_dim, input_dim, fig, plot)

def clear(fig, plot, log):
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
    ui.notify(f'Uploaded {e.name}')
    df = pd.read_excel(io.BytesIO(e.content.read()))[['Date_Time','Volume', 'Price', 'Signed_Volume', "Trading_Time_Indicator"]]
    global global_data
    global_data = df[df["Trading_Time_Indicator"] == "In"]

def perform_clustering(volume: bool, price: bool, signed_volume: bool, datalength: int, price_plot: bool, volume_plot: bool, signed_volume_plot: bool, path_transform: str, clustering: str, sensitivity: float, reservoir_dim: int, input_dim: int, fig, plot):
    # load the selected file 
    parameters = []
    if volume.value == True:
        parameters.append('Volume')
    if price.value == True:
        parameters.append('Price')
    if signed_volume.value == True:
        parameters.append('Signed_Volume')

    try:
        data = global_data[parameters].to_numpy()[:int(datalength.value*global_data.shape[0]/100)]
    except:
        raise Exception('No file uploaded: Please upload a correct input file')
        return
    t = global_data['Date_Time'].to_numpy()[:int(datalength.value*global_data.shape[0]/100)]
    t = pd.to_datetime(t)
    # apply the path transform

    if path_transform.value == 'Lead Lag':
        data = p.lead_lag_transform(data)
    elif path_transform.value == 'Interpolation':
        data = p.interpolation(data)
    # apply the random signature
    if clustering.value == 'Isolation Forest':
        random_sig = rs.RandomSig(reservoir_dim=reservoir_dim.value,input_dim=input_dim.value, anomaly=rs.IForest(contamination=sensitivity.value))
    elif clustering.value == 'LOF':
        random_sig = rs.RandomSig(reservoir_dim=reservoir_dim.value,input_dim=input_dim.value, anomaly=rs.LOF(contamination=sensitivity.value))
    
    labels = random_sig.fit(data)

    df = pd.DataFrame(labels, columns=['Labels_numeric'])
    df['Labels'] = df['Labels_numeric'].apply(lambda x: 'seagreen' if x == -1 else 'royalblue')

    # plot the results
    print('Creating Plot')
    if price_plot.value:
        if input_dim.value == 1:
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,0], mode='markers', marker_color=df['Labels']))
        elif input_dim.value == 2 and volume.value and path_transform.value == 'Lead Lag':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,3], mode='markers', marker_color=df['Labels']))
        elif input_dim.value == 2 and volume.value and path_transform.value == 'Interpolation':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,1], mode='markers', marker_color=df['Labels']))
        else:
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,0], mode='markers', marker_color=df['Labels']))
    if volume_plot.value:
        fig.add_trace(go.Scatter(x=t[500:], y=data[500:,0], mode='markers', marker_color=df['Labels']))
    if signed_volume_plot.value:
        if input_dim.value == 1:
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,0], mode='markers', marker_color=df['Labels']))
        elif input_dim.value == 2 and path_transform.value == 'Lead Lag':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,3], mode='markers', marker_color=df['Labels']))
        elif input_dim.value == 2 and path_transform.value == 'Interpolation':
            fig.add_trace(go.Scatter(x=t[500:], y=data[500:,1], mode='markers', marker_color=df['Labels']))
    plot.update()
