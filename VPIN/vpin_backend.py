# from typing import Union

# import os
# os.environ['R_HOME'] = r"C:\Program Files\R\R-4.3.1"
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# import rpy2
# import rpy2.robjects as ro
# import numpy as np
# import PySimpleGUI as sg
# import subprocess
# import sys
# import asyncio
# from nicegui import ui, events
# import io

# import pandas as pd
# ps = importr('PINstimation')


# def adjpin_func(data: pd.DataFrame, method: str = "ECM", initialsets: str = "GE", num_init: int = 20, restricted: list = [], verbose: bool = True) -> float:
#     """
#     Estimate PIN using the ps.adjpin function from the PINstimation package in R.

#     Args:
#         data (pd.DataFrame): The input data containing the 'Buys' and 'Sells' columns.
#         method (str, optional): The method to be used for PIN estimation. Default is "ECM".
#         initialsets (str, optional): The initial sets method to be used. Default is "GE".
#         num_init (int, optional): The number of initial sets to be used. Default is 20.
#         restricted (list, optional): A list of restricted variables. Default is an empty list.
#         verbose (bool, optional): Whether to print verbose output. Default is True.

#     Returns:
#         float: The PIN estimation value.
#     """
#     # Extract 'Buys' and 'Sells' columns from the input data
#     adjpin_input = data[['Buys', 'Sells']]

#     # Convert the extracted data to an R object
#     with ro.default_converter + pandas2ri.converter:
#         r_adjpin_input = ro.conversion.get_conversion().py2rpy(adjpin_input)

#     return ps.adjpin(r_adjpin_input, method=method, initialsets=initialsets, num_init=num_init, restricted=restricted, verbose=verbose)


# def pin_func(dataset: pd.DataFrame, initialsets: list = [0.3, 0.1, 800, 300, 200], factorization: str = "E", verbose: bool = True) -> list:
#     """
#     Perform PIN estimation using the PINstimation package in R.

#     Args:
#         dataset (pd.DataFrame): A pandas DataFrame containing the 'Buys' and 'Sells' columns representing the trading data.
#         initialsets (list, optional): A list of initial parameter values for the PIN estimation. Defaults to [0.3, 0.1, 800, 300, 200].
#         factorization (str, optional): A string specifying the factorization method to be used in the PIN estimation. Defaults to "E".
#         verbose (bool, optional): Whether to display verbose output during the estimation. Defaults to True.

#     Returns:
#         list: The PIN estimation result.
#     """
#     pin_input = dataset[["Buys", "Sells"]]
#     r_pin_input = pandas2ri.py2rpy(pin_input)
#     init = pandas2ri.py2rpy(pd.DataFrame(initialsets).T)
#     return ps.pin(r_pin_input, init, factorization, verbose)


# def vpin_func(dataset: pd.DataFrame, timebarsize: int = 60, buckets: int = 50, samplength: int = 50,
#               tradinghours: int = 24, verbose: bool = True) -> list:
#     """
#     Estimates Volume-synchronized Probability of Informed Trading (VPIN) using the PINstimation package in R.

#     Args:
#         dataset (pd.DataFrame): A pandas DataFrame containing the 'Date_Time', 'Price', and 'Volume' columns representing the trading data.
#         timebarsize (int, optional): The size of time bars in minutes. Default is 60.
#         buckets (int, optional): The number of buckets to divide the time bars into. Default is 50.
#         samplength (int, optional): The length of the sample used for VPIN estimation. Default is 50.
#         tradinghours (int, optional): The number of trading hours in a day. Default is 24.
#         verbose (bool, optional): Whether to display verbose output during the estimation. Default is True.

#     Returns:
#         list: The VPIN estimation result, which is a list of VPIN values for each time bar.
#     """
#     # Extract the 'Date_Time', 'Price', and 'Volume' columns from the input dataset
#     vpin_input = dataset[['Date_Time', 'Price', 'Volume']]

#     # Convert dataframe to R object
#     with ro.default_converter + pandas2ri.converter:
#         r_vpin_input = ro.conversion.get_conversion().py2rpy(vpin_input)

#     # Import the PINstimation package in R
#     ps = importr('PINstimation')

#     # Call the `ps.vpin` function from the PINstimation package in R, passing the converted data and the specified parameters
#     result = ps.vpin(r_vpin_input, timebarsize, buckets, samplength, tradinghours, verbose)

#     return list(result)

# def handle_upload(e: events.UploadEventArguments) -> None:
#     """
#     Handle the upload event triggered by the user.
    
#     Args:
#         e (events.UploadEventArguments): An object containing information about the uploaded file, including the file name and content.
    
#     Returns:
#         None. The function only assigns the uploaded file data to the appropriate global variables.
#     """
#     ui.notify(f'Uploaded {e.name}')
    
#     file_content = io.BytesIO(e.content.read())
    
#     if 'clean' in e.name:
#         global clean_data
#         clean_data = pd.read_excel(file_content)
#     elif 'bucket' in e.name:
#         global buckets_data
#         buckets_data = pd.read_excel(file_content)
        
# async def button_callback(adjpin, ap_method, ap_initialsets, ap_num_init, pin, p_initialsets, p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s, p_factorization, vpin, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours):
#     """Called after button click"""
#     await asyncio.to_thread(call_functions, adjpin, ap_method, ap_initialsets, ap_num_init, pin, p_initialsets, p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s, p_factorization, vpin, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours)

# def call_functions(adjpin: bool, ap_method: str, ap_initialsets: str, ap_num_init: int, pin: bool, p_initialsets: list, p_alpha: float, p_delta: float, p_mu: float, p_epsilon_b: float, p_epsilon_s: float, p_factorization: str, vpin: bool, vp_time_bars: int, vp_buckets: int, vp_sample_length: int, vp_trading_hours: int) -> None:
#     """
#     Calls three different functions (adjpin_func, pin_func, and vpin_func) and writes their output to a file.

#     Args:
#         adjpin (bool): Whether to call the adjpin_func function.
#         ap_method (str): The method to be used for PIN estimation in adjpin_func.
#         ap_initialsets (str): The initial sets method to be used in adjpin_func.
#         ap_num_init (int): The number of initial sets to be used in adjpin_func.
#         pin (bool): Whether to call the pin_func function.
#         p_initialsets (list): A list of initial parameter values for the PIN estimation in pin_func.
#         p_alpha (float): The alpha parameter for the PIN estimation in pin_func.
#         p_delta (float): The delta parameter for the PIN estimation in pin_func.
#         p_mu (float): The mu parameter for the PIN estimation in pin_func.
#         p_epsilon_b (float): The epsilon_b parameter for the PIN estimation in pin_func.
#         p_epsilon_s (float): The epsilon_s parameter for the PIN estimation in pin_func.
#         p_factorization (str): The factorization method to be used in pin_func.
#         vpin (bool): Whether to call the vpin_func function.
#         vp_time_bars (int): The size of time bars in minutes for VPIN estimation in vpin_func.
#         vp_buckets (int): The number of buckets to divide the time bars into for VPIN estimation in vpin_func.
#         vp_sample_length (int): The length of the sample used for VPIN estimation in vpin_func.
#         vp_trading_hours (int): The number of trading hours in a day for VPIN estimation in vpin_func.

#     Returns:
#         None
#     """
#     global clean_data
#     global buckets_data

#     adjpin_data = ""
#     pin_data = ""
#     vpin_data = ""

#     if adjpin:
#         try:
#             adjpin_data = str(adjpin_func(buckets_data, ap_method, ap_initialsets, ap_num_init))
#         except Exception as e:
#             print(e)
    
#     if pin:
#         try:
#             pin_data = str(pin_func(buckets_data, [p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s], p_factorization))
#         except Exception as e:
#             print(e)

#     if vpin:
#         try:
#             vpin_data = str(vpin_func(clean_data, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours))
#         except Exception as e:
#             print(e)

#     filename = clean_data.columns[0].split()[0] + "_pin_output.txt"
#     with open(filename, "w") as f:
#         f.write(adjpin_data)
#         f.write(pin_data)
#         f.write(vpin_data)
    