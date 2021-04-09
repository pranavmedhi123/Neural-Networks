# Medhi, Pranav
# 1001-756-326
# 2020-02-29
# Assignment-02-01

import numpy as np
import math


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.number_of_nodes=number_of_nodes
        self.transfer_function=transfer_function
        self.input_dimensions=input_dimensions

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed!=None:
            np.random.seed(seed)
        self.weights=np.random.randn(self.number_of_nodes,self.input_dimensions)
        

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if W.shape!=self.weights.shape:
            return -1
        self.weights=W

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        if self.transfer_function=="Hard_limit":
            product=np.dot(self.weights,X)
            product[product>=0]=1
            product[product<0]=0
            return product
        else:
            product=np.dot(self.weights,X)
            return product

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        P_pseudo=np.linalg.pinv(X)
        Weight_new=np.dot(y,P_pseudo)
        self.weights=Weight_new

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        for x in range(0,num_epochs):
            k=0
            
            for i in range(0, int(math.ceil(X.shape[1])/batch_size)):
                input_batch=X[:,k:k+batch_size]
                input_Transpose=input_batch.transpose()
                
                #print(input_batch)
                y_batch=y[:,k:k+batch_size]

                output=self.predict(input_batch)
                
                Actual_output=np.subtract(y_batch,output)
                k+=batch_size
                
                
                
                if learning=="Filtered":
                    
                    self.weights=(1-gamma)*self.weights+np.dot(y_batch,input_Transpose)*alpha
                    #self.weights=new_weight
                    
                elif learning=="Delta" or learning == "delta":
                    
                    self.weights=self.weights+np.dot(Actual_output,input_Transpose)*alpha
                    #self.weights=new_weight
                else:
                    self.weights=self.weights+np.dot(output,input_Transpose)*alpha

            

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        A=self.predict(X) 
        error= (np.square(np.subtract(y, A))).mean(None)
        return error

    
    



