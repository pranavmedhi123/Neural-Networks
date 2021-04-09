# Medhi, Pranav
# 1001-756-326
# 2020-02-16
# Assignment-01-01



import numpy as np
class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize Perceptron model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        
        self.number_of_nodes=number_of_nodes
        self.input_dimensions=input_dimensions
        self.initialize_weights()
        
        
        
    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        
        
        if seed != None:
            np.random.seed(seed)
            
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions+1)
            

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        
        self.weights = W
        

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on a batch of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
#         for n in range(0,number_of_nodes):
        one_array=np.ones((X.shape[1]))
#         print('pred')
    
        
        k=np.insert(X,0,one_array,axis=0)
#         print(k)
        product=np.dot(self.weights,k)
        product[product>0]=1
        product[product<0]=0
        return product
            
            
            


    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        for x in range(0,num_epochs):
            for i in range(0,len(X[0])):
                p=[]
                target = []
                for j in range(0,len(X)):
                    p.append(X[j][i])
                for j in range(0,len(Y)):
                    target.append(Y[j][i])
                p.insert(0,1)
                p_Transpose=np.array(p)
                #p_Transpose=p_Transpose.transpose()
                p_Transpose_mat=p_Transpose [np.newaxis]
                #print(p_Transpose_mat.shape)
        #             p_Transpose_mat=p_Transpose_mat.transpose()
                #print(p_Transpose_mat.shape)
                output= np.dot(self.weights,p_Transpose)
                output[output>=0]=1
                output[output<0]=0




    #             for i in range(0,len(X[el])):
    #                 target=[]
    #                 for j in range(0,number_of_nodes):
    #                     target.append(Y[j][i])


                target_Transpose= np.array(target)
                target_Transpose=target_Transpose.transpose()

                error=np.subtract(target_Transpose,output)
                error_matrix= error [np.newaxis]
                error_matrix_Transpose=error_matrix.transpose()
        #         print(error_matrix_Transpose)
                #print(';akjsdflkasjdfalskdjfas;lkdk')
#                 print(np.dot(error_matrix_Transpose,p_Transpose_mat))
                new_weight=self.weights+np.dot(error_matrix_Transpose,p_Transpose_mat)*alpha
                
                self.weights = new_weight

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """

        output = self.predict(X)
        number_of_element = len(X[0])
        correct_counter=0
        compare = np.asarray(output)==Y
#         print(compare)
#         print(compare[:,1])
        for i in range(number_of_element):
            if compare[:,i].all():
                correct_counter +=1
#                 print('error')
            
        return 100*(number_of_element-correct_counter)/number_of_element
     
        


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())