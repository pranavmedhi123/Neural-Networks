# Medhi, Pranav
# 1001-756-326
# 2020_03_22
# Assignment-03-01

# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import math

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimensions=input_dimension
        self.nn = []
        self.weights = []
        

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.num_nodes=num_nodes
        if not self.nn:
            weight_matrix=np.random.randn(self.input_dimensions,self.num_nodes)
#             self.transfer_function=transfer_function
            
        else:
            weight_matrix=np.random.randn(self.nn[-1]['weight'].shape[1],self.num_nodes)
#             self.transfer_function=transfer_function
        bias=np.random.randn(num_nodes)

        layer={
            'weight':weight_matrix,
            'transfer':transfer_function,
            'bias':bias
            }
        self.weights.append(None)
        self.nn.append(layer)
        

        

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.nn[layer_number]["weight"]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.nn[layer_number]['bias']

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.nn[layer_number]['weight']=weights
        

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.nn[layer_number]['bias']=biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat, name=None))
        
        

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        inp=X
        for i in range(0,len(self.nn)):
            inp=tf.matmul(inp,self.nn[i]['weight']) 
            inp=tf.add(inp,self.nn[i]['bias'])

            if self.nn[i]['transfer'].lower()=='linear':
                inp = inp
#                 print('linear')
            elif self.nn[i]['transfer'].lower()=='relu':
#                 print('relu')
                inp=tf.nn.relu(inp)
            elif self.nn[i]['transfer'].lower()=='sigmoid':
#                 print('sigmoid')
                inp=tf.nn.sigmoid(inp)
        return inp
            
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """

        for i in range(0,num_epochs):
            for j in range(0, len(X_train), batch_size):


                
                input_batch = X_train[j:j+batch_size]
                output_batch = y_train[j:j+batch_size]

                
                all_weights = []
                all_bias = []
                for i in range (len(self.nn)):
                    all_weights.append(self.nn[i]['weight'])
                    all_bias.append(self.nn[i]['bias'])
                

                with tf.GradientTape() as tape:
                    y_hat = self.predict(input_batch)
                    loss = self.calculate_loss(output_batch, y_hat)
                    # Note that `tape.gradient` works with a list as well (w, b).
                    dloss_dw, dloss_db = tape.gradient(loss, [all_weights, all_bias])
                for k in range(len(all_weights)):
                    all_weights[k].assign_sub(alpha * dloss_dw[k])
                    all_bias[k].assign_sub(alpha * dloss_db[k])
                
                
                
                
    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        counter = 0
        desired = y
        actual = self.predict(X)
        actual = np.argmax(actual, axis =1)
        for i in range (len(actual)):
            if desired[i] != actual[i]:
                counter+=1
        return (counter/len(actual))


        

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        desired = y
        actual = self.predict(X)
        actual = np.argmax(actual,axis=1)
        return tf.math.confusion_matrix(desired,actual)
