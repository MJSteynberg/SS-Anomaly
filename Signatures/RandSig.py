from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import sklearn
from sklearn.ensemble import IsolationForest as IForest
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class RandomSignature(ABC):
    """
    Abstract class for random signatures
    """

    @abstractmethod
    def _forward(self):
        """
        Performs the forward pass of the random signature method.

        Returns:
            The next state value of the random signature.
        """
        pass

    @abstractmethod
    def _obtain_signatures(self):
        """
        Obtains the entire forward pass through the dataset.

        Returns:
            A collection of all signatures for the length of the input data.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Obtains the signatures using the `_obtain_signatures()` method and fits them to the chosen anomaly detection algorithm.

        Returns:
            The anomaly score of each signature.
        """
        pass

class RandomSig(RandomSignature):  
    def __init__(self, reservoir_dim: int, input_dim: int, act: str = 'tanh', anomaly = IForest(),
                 reservoir_std: float = 0.15, contamination: float = 'auto', random_state: int = None, discard: int = 500):
        """
        Initialization of the RandomSig class

        Args:
            reservoir_dim (int): dimension of reservoir for random signature
            input_dim (int): dimension of input data
            act (string): The chosen activation (tanh) # Placeholder for future implementation
            anomaly (sklearn.ensemble or sklearn.neighbors, optional): The anomaly detection algorithm needed. Defaults to IsolationForest().
            reservoir_std (float, optional): Standard deviation of reservoir matrix. Defaults to 0.15.
            contamination (float, optional): Contamination of anomalies. Defaults to the method in the original paper.
            random_state (int, optional): Controls the random state of the system. Defaults to None.
            discard (int, optional): Number of points to use to warm up the reservoir. Defaults to 5000.
        """
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.contamination = contamination
        self.random_state = random_state
        self.reservoir = [np.random.normal(0, reservoir_std, size=(reservoir_dim, reservoir_dim)) for _ in range(input_dim)]
        self.bias = [np.random.normal(0, 1, size=(reservoir_dim, 1)) for _ in range(input_dim)]
        self.reservoir = np.array(self.reservoir)
        self.bias = np.array(self.bias)
        self.anomaly = anomaly
        self._activation = np.tanh
        self.discard = discard
        self.scaler = StandardScaler()
        try:
            self.anomaly.contamination = contamination
        except AttributeError:
            print("No contamination parameter needed")
        try: 
            self.anomaly.random_state = random_state
        except AttributeError: 
            print("No random state parameter needed")
        try:
            self.anomaly.verbose = 1
        except AttributeError:
            print("No verbose parameter needed")
    
    def _forward(self, input0, input1, state):
        """
        Calculates the next state value of the random signature based on the current state, two input arrays,
        and the reservoir and bias matrices.

        Args:
            input0 (np.array): The first input array of the bi-input of the random signature method.
            input1 (np.array): The second input array of the bi-input of the random signature method.
            state (np.array): The current state value of the random signature.

        Returns:
            np.array: The next state value of the random signature.
        """
        next_state = state.copy()  # Create a copy of the current state

        for i in range(self.input_dim):
            diff = input1[i] - input0[i]  # Calculate the difference between corresponding elements of input1 and input0
            activation = self._activation(np.matmul(self.reservoir[i], state) + self.bias[i].reshape(self.reservoir_dim))
            next_state += activation * diff  # Update the next state value

        return next_state
    
    def _obtain_signatures(self, X: np.ndarray, N: int, Z0: np.ndarray) -> np.ndarray:
        """
        Obtain the entire forward pass through the dataset.

        Args:
            X (np.ndarray): The "control" or "input sequence" for which the random signatures are obtained.
            N (int): Length of sequence of input data.
            Z0 (np.ndarray): Initial signature or state value for all passes.

        Returns:
            np.ndarray: Collection of all signatures for the length of input data N.
        """
        self.scaler.fit_transform(X)
        Z = np.zeros((N - self.discard, self.reservoir_dim))
        Z[0] = Z0

        for i in tqdm(range(1, N), desc="Obtaining Signatures", unit="iteration", miniters=10):
            if i > self.discard:
                Z[i - self.discard] = self._forward(X[i - 1], X[i], Z[i - 1 - self.discard])
            elif i == self.discard:
                Z[0] = self._forward(X[i - 1], X[i], Z[self.discard - 1])
            else:
                Z[i] = self._forward(X[i - 1], X[i], Z[i - 1])

        return Z
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Obtain the signatures and fit them to the chosen anomaly detection algorithm

        Args:
            X (np.ndarray): The sequence of "control" or "input" data

        Returns:
            np.ndarray: The anomaly score of each signature
        """
        N = X.shape[0]
        Z0 = np.random.normal(0, 1, size=(self.reservoir_dim))
        Z = self._obtain_signatures(X, N, Z0)
        print("Fitting the anomaly detection algorithm")
        return self.anomaly.fit_predict(Z.reshape(N - self.discard, self.reservoir_dim))