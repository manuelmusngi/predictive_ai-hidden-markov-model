import numpy as np
from pyhhmm.gaussian import GaussianHMM

class HMMModel:
    def __init__(self, n_states, covariance_type, n_emissions):
        self.model = GaussianHMM(n_states=n_states, covariance_type=covariance_type, n_emissions=n_emissions)

    def train(self, X_train):
        self.model.train([np.array(X_train)])
    
    def predict(self, X_test):
        return self.model.predict([np.array(X_test)])[0]
