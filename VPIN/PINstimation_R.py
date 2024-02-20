
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


#Create a gui which accepts two files: The clean and buckets files, and a choice to run each of the methods on that data 
import os.path

# Create gui so that user can select file, and enter all parameters
sg.theme('DarkBlue')   # Add a touch of color
wid = 61
wid2 = 8
layout = [ [sg.Column([[sg.Frame(title = "Datasets", layout = [[ sg.Column([[sg.Text("")], [sg.Text('Clean File: ', size=(wid-10, 1)), sg.Input(), sg.FileBrowse()],           
            [sg.Text('Buckets File: ', size=(wid-10, 1)), sg.Input(), sg.FileBrowse()]])]])],
            [sg.Frame(title = "AdjPIN",layout=[[sg.Checkbox('Select', default=True)],[sg.Text('Method', size=(wid, 1)), sg.InputText("ECM")],[sg.Text('Initialsets', size=(wid, 1)), sg.InputText("GE")],[sg.Text('Num_Init', size=(wid, 1)), sg.InputText("20")]])],
            [sg.Frame(title = "PIN",layout=[[sg.Checkbox('Select', default=True)],[sg.Text('Initialsets:', size=(10, 1)), sg.Text('alpha=', size=(wid2, 1)), sg.InputText("0.3", size=(wid2, 1)), sg.Text('delta=', size=(wid2, 1)), sg.InputText("0.1", size=(wid2, 1)), sg.Text('mu=', size=(wid2, 1)), sg.InputText("800", size=(wid2, 1)), sg.Text('epsilon_b=', size=(wid2, 1)), sg.InputText("300", size=(wid2, 1)), sg.Text('epsilon_s=', size=(wid2, 1)), sg.InputText("200", size=(wid2, 1))],[sg.Text('Factorization', size=(wid, 1)), sg.InputText("E")]])],
            [sg.Frame(title = "VPIN",layout=[[sg.Checkbox('Select', default=True)],[sg.Text('Time Bars', size=(wid, 1)), sg.InputText("60")],[sg.Text('Buckets', size=(wid, 1)), sg.InputText("50")], [sg.Text('Sample Length', size=(wid, 1)), sg.InputText("50")], [sg.Text('Trading Hours', size=(wid, 1)), sg.InputText("24")]])],
            [sg.Submit(), sg.Cancel(), sg.Text('', size=(48, 1)) , sg.CloseButton("Close")]]),
            ]]


           
# Create the Window
window = sg.Window('Data Cleaning', layout, size = (1200, 700))
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):   # if user closes window or clicks cancel
        raise Exception("User cancelled")
        break
    if event == 'Submit':
        #Get the values from the gui
        clean_path = values[0]
        bucket_path = values[1]
        adjpin_val = values[2]
        adjpin_method = values[3]
        adjpin_initialsets = values[4]
        adjpin_num_init = int(values[5])
        pin_val = values[6]
        pin_alpha = float(values[7])
        pin_delta = float(values[8])
        pin_mu = float(values[9])
        pin_epsilon_b = float(values[10])
        pin_epsilon_s = float(values[11])
        pin_factorization = values[12]
        vpin_val = values[13]
        vpin_timebars = int(values[14])
        vpin_buckets = int(values[15])
        vpin_samplelength = int(values[16])
        vpin_tradinghours = int(values[17])
        ######################
        # Parameters #
        ######################
        #Import data
        try:
            Clean_Data = pd.read_excel(clean_path)
        except:
            raise Exception("Clean file not found")
        
        try:
            Buckets_Data = pd.read_excel(bucket_path)
        except:
            raise Exception("Buckets file not found")
        sg.Print("Data imported successfully")
        f = open(f"{clean_path.split('_')[0]}_{clean_path.split('_')[1]}_PIN_Output.txt", "w")

        if adjpin_val:
            try:
                sg.Print("AdjPIN Calculating...")
                adjpin_data = str(adjpin(Buckets_Data, adjpin_method, adjpin_initialsets, adjpin_num_init))
                f.write(adjpin_data)
                sg.Print("AdjPIN Completed")

            except:
                raise Exception("PIN Failed")
            
            
        if pin_val:
            try:
                sg.Print("PIN Calculating...")
                pin_data = str(pin(Buckets_Data, [pin_alpha, pin_delta, pin_mu, pin_epsilon_b, pin_epsilon_s], pin_factorization))
                f.write(pin_data)
                sg.Print("PIN Completed")
            except:
                raise Exception("PIN Failed")
            
        if vpin_val:
            try:
                sg.Print("VPIN Calculating...")
                vpin_data = str(vpin(Clean_Data, vpin_timebars, vpin_buckets, vpin_samplelength, vpin_tradinghours))
                f.write(vpin_data)
                sg.Print("VPIN Completed")
            except:
                raise Exception("VPIN Failed")
            
        f.close()
        sg.Print("Output file created")

    if event == "Close":
        break



