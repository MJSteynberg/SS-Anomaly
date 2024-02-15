import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d, interp2d

def lead_lag_transform(data: np.ndarray) -> np.ndarray:
    """
    Perform a lead-lag transformation on the input data as defined in: 
        Flint, G., Hambly, B. and Lyons, T. (2016) 
        Discretely sampled signals and the rough Hoff process, arXiv.org. 
        Available at: https://doi.org/10.48550/arXiv.1310.4054 (Accessed: 19 January 2024). 

    Args:
        data (numpy.ndarray): The input data to be transformed. It can be a 1D array or a 2D array.

    Returns:
        numpy.ndarray: The transformed array with twice the number of columns as the input array.
        The first half of the columns represent the lead values and the second half represent the lag values.
    """
    #Error handling
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except:
            raise TypeError("Input data must be a numpy array or a list.")

    #Perform the lead-lag transformation    
    n, dim = data.shape if data.ndim == 2 else (len(data), 1)
    t = np.linspace(0, 1, n)
    lead_lag = np.zeros((n, 2 * dim))

    for k in range(n - 1):
        lead_lag[k, :dim] = data[k]
        lead_lag[k, dim:] = data[k + 1]
        
    lead_lag[-1, :dim] = data[-1]
    lead_lag[-1, dim:] = data[-1]

    return lead_lag
def interpolation(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Applies rectilinear interpolation to a given list or array.

    Args:
        data (np.ndarray): The input data.
        method (str, optional): The interpolation method to use. Defaults to 'linear'.

    Returns:
        np.ndarray: The transformed data.
    """
    if data.size == 0:
        return np.array([])
    
    dim = data.ndim
    n = data.shape[0]
    
    if dim == 1:
        t = np.linspace(0, 1, n)
        f = interp1d(t, data, kind=method)
        return f(np.linspace(0, 1, n))
    else:
        t = np.linspace(0, 1, n)
        x, y = np.meshgrid(t, np.arange(dim))
        f = interp2d(x, y, data.T, kind=method)
        return f(np.linspace(0, 1, n), np.arange(dim)).T
    
