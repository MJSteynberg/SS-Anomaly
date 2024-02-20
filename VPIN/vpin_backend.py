
import os
os.environ['R_HOME'] = r"C:\Program Files\R\R-4.3.1"
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2
import rpy2.robjects as ro
import numpy as np
import PySimpleGUI as sg
import subprocess
import sys
import asyncio
from nicegui import ui, events
import io

import pandas as pd
ps = importr('PINstimation')


def adjpin(data, method = "ECM", initialsets = "GE", num_init = 20, restricted = list(), verbose = True):
    adjpin_input = data[["Buys", "Sells"]]
    with (ro.default_converter + pandas2ri.converter).context():
        r_adjpin_input = ro.conversion.get_conversion().py2rpy(adjpin_input)
    return ps.adjpin(r_adjpin_input, method, initialsets, num_init, restricted, verbose)

def mpin_ecm(dataset):
    mpin_input = dataset[["Buys", "Sells"]]
    with (ro.default_converter + pandas2ri.converter).context():
        r_mpin_input = ro.conversion.get_conversion().py2rpy(mpin_input)
    return ps.mpin_ecm(r_mpin_input)

def mpin_ml(dataset):
    mpin_input = dataset[["Buys", "Sells"]]
    with (ro.default_converter + pandas2ri.converter).context():
        r_mpin_input = ro.conversion.get_conversion().py2rpy(mpin_input)
    return ps.mpin_ml(r_mpin_input)


def pin(dataset,  initialsets = [0.3, 0.1, 800, 300, 200], factorization = "E", verbose = True):
    pin_input = dataset[["Buys", "Sells"]]
    #Convert the dataframe to r
    with (ro.default_converter + pandas2ri.converter).context():
        r_pin_input = ro.conversion.get_conversion().py2rpy(pin_input)
        init = ro.conversion.get_conversion().py2rpy(pd.DataFrame(initialsets).T)
    return ps.pin(r_pin_input, init, factorization, verbose)

def pin_bayes(dataset):
    pin_input = dataset[["Buys", "Sells"]]
    #Convert the dataframe to r
    with (ro.default_converter + pandas2ri.converter).context():
        r_pin_input = ro.conversion.get_conversion().py2rpy(pin_input)
    return ps.pin(r_pin_input)

def pin_ea(dataset):
    pin_input = dataset[["Buys", "Sells"]]
    #Convert the dataframe to r
    with (ro.default_converter + pandas2ri.converter).context():
        r_pin_input = ro.conversion.get_conversion().py2rpy(pin_input)
    return ps.pin(r_pin_input)

def pin_gwj(dataset):
    pin_input = dataset[["Buys", "Sells"]]
    #Convert the dataframe to r
    with (ro.default_converter + pandas2ri.converter).context():
        r_pin_input = ro.conversion.get_conversion().py2rpy(pin_input)
    return ps.pin(r_pin_input)

def pin_yz(dataset):
    pin_input = dataset[["Buys", "Sells"]]
    #Convert the dataframe to r
    with (ro.default_converter + pandas2ri.converter).context():
        r_pin_input = ro.conversion.get_conversion().py2rpy(pin_input)
    return ps.pin(r_pin_input)


def vpin(dataset, timebarsize = 60, buckets = 50, samplength = 50, tradinghours = 24, verbose = True):
    vpin_input = dataset[["Date_Time", "Price", "Volume"]]
    #Convert dataframe to r
    with (ro.default_converter + pandas2ri.converter).context():
        r_vpin_input = ro.conversion.get_conversion().py2rpy(vpin_input)
    
    return ps.vpin(r_vpin_input, timebarsize, buckets, samplength, tradinghours, verbose)

def handle_upload(e: events.UploadEventArguments):
    ui.notify(f'Uploaded {e.name}')
    df = pd.read_excel(io.BytesIO(e.content.read()))
    print(df.head())
async def button_callback(adjpin, ap_method, ap_initialsets, ap_num_init, pin, p_initialsets, p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s, p_factorization, vpin, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours):
    """Called after button click"""
    await asyncio.to_thread(call_functions, adjpin, ap_method, ap_initialsets, ap_num_init, pin, p_initialsets, p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s, p_factorization, vpin, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours)

def call_functions(adjpin, ap_method, ap_initialsets, ap_num_init, pin, p_initialsets, p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s, p_factorization, vpin, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours):
    pass