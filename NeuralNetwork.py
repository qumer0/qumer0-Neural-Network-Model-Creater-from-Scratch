import numpy as np
from datetime import datetime
import ast
import os
from typing import Union
import matplotlib.pyplot as plt
from pooling import MaxPooling2D,MinPooling2D,AvaragePooling2D,MaxUnPool2D,MinUnPool2D,AvarageUnPool2D

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim) * 0.01
        self.bias = np.zeros((output_dim, 1))
        self.params = output_dim * input_dim + output_dim
    
    def forward(self, input) ->list:
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate) ->list:
        # Calculation of the weights gradient
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        # Calculation of the input gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Calculation of the new weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        return input_gradient
    
class ActivationLayer:
    def __init__(self,func_name) -> None:
        self.function_name = func_name    
        self.params = 0
      
    def __sigmoid__(self,x):
        x = np.clip(x, -500, 500) 
        return 1/(1 + np.exp(-x))
    
    def __tanh__(self,x):
        return np.tanh(x)

    def __relu__(self,x):
        return np.maximum(0, x)
    

    def __softmax__(self,x):
        exps = np.exp(x - np.max(x)) 
        return exps / np.sum(exps, axis=0)


    def __act_func__(self,function_name,x):
        if function_name == "sigmoid":
            return self.__sigmoid__(x)
        elif function_name =="softmax":
            return self.__softmax__(x)
        elif function_name=="tanh":
            return self.__tanh__(x)
        elif function_name == "relu":
            return self.__relu__(x)
        else:
            raise ValueError("Please make sure the activations function name is corret syntax.") 
        
    def forward(self,input) -> list:
        """
        Calculate the Activation functions for given input
        """

        self.input = input
        return self.__act_func__(self.function_name,self.input)    
        
    def __sigmoid_deriative__(self,x):
        """
        Deriative of the sigmoid functions

        """
        return self.__sigmoid__(x) * (1- self.__sigmoid__(x))

    # def __softmax_derivative__(self,softmax_output):
    #     """
    #     Deriative of the softmax function

    #     """
        
    #     s = softmax_output.reshape(-1,1)
    #     return np.diagflat(s) - np.dot(s, s.T)


    def __softmax_derivative__(self,input,output):
        """
        Deriative of the softmax function

        """
        
        n = np.size(output)
        return np.dot((np.identity(n) - output.T) * output, input)

    def __tanh_derivative__(self,x):
        """
        Deriative of the tanh function

        """
        return 1 - np.tanh(x)**2


    def __relu_derivative__(self,x):
        """
        Deriative of the relu function

        """
        return np.where(x > 0, 1, 0)


    def __deriative_act_func__(self,function_name,x,output):
        if function_name == "sigmoid":
            return self.__sigmoid_deriative__(x)
        elif function_name =="softmax":
            return self.__softmax_derivative__(x,output)
        elif function_name=="tanh":
            return self.__tanh_derivative__(x)
        elif function_name == "relu":
            return self.__relu_derivative__(x)
        else:
            raise ValueError("Please make sure the activations function name is corret syntax.")    

    

    def backward(self, output_gradient, learning_rate) -> list:
        """
        Multiply output gradient and deriative of functions
        """
        return np.multiply(output_gradient,self.__deriative_act_func__(self.function_name,self.input,self.forward(self.input)))

# class ConvolutionalLayer:
#     def __init__(self,input_shape,kernel_size,kernel_count,stride:int=1) -> None:
        
#         self.stride = stride
#         self.input_shape = input_shape
#         self.input_with,self.inputh_height,self.input_channel = input_shape
#         self.kernel_count = kernel_count
#         self.kernel_with , self.kernel_height = kernel_size , kernel_size
#         self.out_with,self.out_height,self.out_channel = int((self.input_with-self.kernel_with)/stride+1),int((self.inputh_height-self.kernel_height)/stride+1),kernel_count
#         self.output_shape = (self.out_with,self.out_height,self.out_channel)
#         self.params = self.kernel_with * self.kernel_height * self.input_channel * kernel_count + self.out_with*self.out_height*self.out_channel
        
#         self.kernel = np.random.randn(self.kernel_with,self.kernel_height,self.input_channel,kernel_count)* 0.01
#         self.bias = np.random.rand(self.out_with,self.out_height,self.out_channel)* 0.01
        

#     def Conv2D(self,matrix1,matrix2,stride=1):
    
#         kernel_with,kernel_height = matrix2.shape
#         input_with,input_height = matrix1.shape
#         out_with,out_height = int((input_with-kernel_with)/stride+1) , int((input_height-kernel_height)/stride+1)
#         output = np.zeros((out_with,out_height))
#         for w in range(out_with):
#             for h in range(out_height):
#                 output[w,h] = np.sum((matrix1[w:kernel_with+w,h:kernel_height+h]) * matrix2)
#         return output

#     def ConvMulty(self,matrix1,kernel_size,gradiant,stride=1):
#         kernel_with,kernel_height = kernel_size,kernel_size
#         input_with,input_height = matrix1.shape
#         out_with,out_height = int((input_with-kernel_with)/stride+1) , int((input_height-kernel_height)/stride+1)
#         output = np.zeros((kernel_with,kernel_height))
    
#         for w in range(out_with):
#             for h in range(out_height):
#                 output = output +  matrix1[w:kernel_with+w,h:kernel_height+h] * gradiant[w,h]
                
#         return output


#     def Convolve(self,input_matrix, kernel):
        
#         input_height, input_width = input_matrix.shape
#         kernel_height, kernel_width = kernel.shape
        
        
#         output_height = input_height + kernel_height - 1
#         output_width = input_width + kernel_width - 1
#         output_matrix = np.zeros((output_height, output_width))
        
        
#         padded_input = np.pad(input_matrix, ((kernel_height - 1, kernel_height - 1),
#                                             (kernel_width - 1, kernel_width - 1)),
#                             mode='constant')
        
        
#         kernel = np.flipud(np.fliplr(kernel))
        
        
#         for i in range(output_height):
#             for j in range(output_width):
#                 output_matrix[i, j] = np.sum(padded_input[i:i+kernel_height, j:j+kernel_width] * kernel)
        
#         return output_matrix




#     def forward(self,inputs):
#         self.inputs = inputs
#         self.output = np.zeros(self.output_shape)
#         for ker in range(self.kernel_count):
#             for ch in range(self.input_channel):
#                 self.output[:,:,ker] += self.Conv2D(inputs[:,:,ch],self.kernel[:,:,ch,ker],self.stride)
        
            
#         self.output = self.output + self.bias
#         return self.output 
    
#     def backward(self,output_gradient,learning_rate):
#         kernels_gradient = np.zeros((self.kernel_with,self.kernel_height,self.input_channel,self.kernel_count))
#         inputs_gradient = np.zeros((self.input_with,self.inputh_height,self.input_channel))       
#         for ker in range(self.kernel_count):
#             for ch in range(self.input_channel):   
#                 kernels_gradient[:,:,ch,ker] = self.ConvMulty(self.inputs[:,:,ch],self.kernel_with,output_gradient[:,:,ker],self.stride)
#                 inputs_gradient[:,:,ch] +=  self.Convolve( output_gradient[:,:,ker] , self.kernel[:,:,ch,ker])
#         self.kernel = self.kernel - learning_rate*kernels_gradient
#         self.bias = self.bias - learning_rate*output_gradient
        
#         return inputs_gradient    
    
class ConvolutionalLayer:
    def __init__(self,input_shape,kernel_size,kernel_count,stride:int=1) -> None:
        
        self.stride = stride
        self.input_shape = input_shape
        self.input_with,self.inputh_height,self.input_channel = input_shape
        self.kernel_count = kernel_count
        self.kernel_with , self.kernel_height = kernel_size , kernel_size
        self.out_with,self.out_height,self.out_channel = int((self.input_with-self.kernel_with)/stride+1),int((self.inputh_height-self.kernel_height)/stride+1),kernel_count
        self.output_shape = (self.out_with,self.out_height,self.out_channel)
        self.params = self.kernel_with * self.kernel_height * self.input_channel * kernel_count + self.out_with*self.out_height*self.out_channel
        
        self.kernel = np.random.randn(self.kernel_with,self.kernel_height,self.input_channel,kernel_count)* 0.01
        self.bias = np.random.rand(self.out_with,self.out_height,self.out_channel)* 0.01
        

    
    def conv2d(self,inputs, weights):
        h, w = weights.shape
        output = np.zeros((inputs.shape[0] - h + 1, inputs.shape[1] - w + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(inputs[i:i+h, j:j+w] * weights)
        return output



    def conv2d_backward_input(self,inputs, weights, output_gradient):
        h, w = weights.shape
        input_gradient = np.zeros_like(inputs)
        
    
        # Gradient w.r.t. input X
        for i in range(output_gradient.shape[0]):
            for j in range(output_gradient.shape[1]):
                input_gradient[i:i+h, j:j+w] += output_gradient[i, j] * weights
                
   
        return input_gradient

    def conv2d_backward_kernel(self,inputs, weights, output_gradient):
        h, w = weights.shape
        
        weights_gradient = np.zeros_like(weights)
    
        # Gradient w.r.t. input X
        for i in range(output_gradient.shape[0]):
            for j in range(output_gradient.shape[1]):
                
                weights_gradient += output_gradient[i, j] * inputs[i:i+h, j:j+w]
   
        return weights_gradient


    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.zeros(self.output_shape)
        for ker in range(self.kernel_count):
            for ch in range(self.input_channel):
                self.output[:,:,ker] += self.conv2d(self.inputs[:,:,ch],self.kernel[:,:,ch,ker])
                
            self.output[:,:,ker] = self.output[:,:,ker] + self.bias[:,:,ker]
        return self.output 
    
    def backward(self,output_gradient,learning_rate):
        kernels_gradient = np.zeros((self.kernel_with,self.kernel_height,self.input_channel,self.kernel_count))
        inputs_gradient = np.zeros((self.input_with,self.inputh_height,self.input_channel))       
        for ker in range(self.kernel_count):
            for ch in range(self.input_channel):   
                kernels_gradient[:,:,ch,ker] = self.conv2d_backward_kernel(self.inputs[:,:,ch],self.kernel[:,:,ch,ker],output_gradient[:,:,ker])
                inputs_gradient[:,:,ch] += self.conv2d_backward_input(self.inputs[:,:,ch],self.kernel[:,:,ch,ker],output_gradient[:,:,ker])
        self.kernel = self.kernel - learning_rate*kernels_gradient
        self.bias = self.bias - learning_rate*output_gradient
        
        return inputs_gradient
    

class Pooling:
    def __init__(self,poolingtype,size:int=2,stride:int=2) -> None:
        self.poolingtype = poolingtype
        self.poolsize = size
        self.stride = stride
        self.params = 0

    def poolingselection(self,poolingtype,input,output_shape,size=2,stride=2,):
        if poolingtype=="MaxPooling":
            return MaxPooling2D(input,size,stride)
        elif poolingtype=="MinPooling":
            return MinPooling2D(input,size,stride)
        elif poolingtype== "AvaragePoling":
            return AvaragePooling2D(input,size,stride)
        else:
            raise ValueError("Please check pooling type. Wrong name or usage")
        

    def unpoolingselection(self,poolingtype,input,max_indices,original_shape,size=2,stride=2,):
        if poolingtype=="MaxPooling":
            return MaxUnPool2D(input,max_indices,original_shape,size,stride)
        elif poolingtype=="MinPooling":
            return MinUnPool2D(input,max_indices,original_shape,size,stride)
        elif poolingtype== "AvaragePoling":
            return AvarageUnPool2D(input,max_indices,original_shape,size,stride)
        else:
            raise ValueError("Please check pooling type. Wrong name or usage")


    def forward(self,input):
        self.input = input
        self.output_shape = (int((input.shape[0] - self.poolsize) // self.stride + 1), int((input.shape[1] - self.poolsize) // self.stride + 1),self.input.shape[2])
        self.indices = np.zeros(self.output_shape,dtype=tuple)
        self.output = np.zeros(self.output_shape)
        for i in range(self.input.shape[2]):
            self.output[:,:,i],self.indices[:,:,i] = self.poolingselection(self.poolingtype,self.input[:,:,i],self.poolsize,self.stride)

        return self.output

    def backward(self,output_gradient,learning_rate):
        output =np.zeros_like(self.input)
        original_shape = (self.input.shape[0],self.input.shape[1])
        for i in range(output.shape[2]):
            output[:,:,i] = self.unpoolingselection(self.poolingtype,output_gradient[:,:,i],self.indices[:,:,i],original_shape,self.poolsize,self.stride)

        return output   

class Flutten:
    def __init__(self,input_shape,output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.params = 0
    def forward(self,input):
        return np.reshape(input,self.output_shape)
    def backward(self,output_gradient,learning_rate):
        return np.reshape(output_gradient,self.input_shape)

class Nn:
    def __init__(self):
        self.layers = []
        self.actfunctions=[]
        self.input_dim = None

    def init_input_count(self, input_count:int):
        """
        input_count -> Number of inputs to the model

        """
        
        self.input_dim = input_count

    def addDense(self, output_dim:int):

        """
        output_dim -> Count of the hidden layer neuron
        
        """
        flut = [layer for layer in self.layers if isinstance(layer, Flutten)]
        fclayers = [layer for layer in self.layers if isinstance(layer, DenseLayer)]

        if len(flut)!=0:
            
            self.input_dim = flut[-1].output_shape[0]
            

        if self.input_dim is None:
            raise ValueError("Input dimension not initialized. Call init_input_count first or flutten the convolutional layers.")
        
        if len(fclayers)== 0:
            input_dim = self.input_dim
        else:
            
            input_dim = fclayers[-1].weights.shape[0]
    

        
        self.layers.append(DenseLayer(input_dim, output_dim))

    def forward(self, X) -> list:
        """
        X is the input. Return prediction in current model.
        """
        for layer in self.layers:
            X = layer.forward(X)
            
        return X

    def backward(self, output_gradient, learning_rate):
        """
        output_gradient -> Gradient of the last layer

        learning_rate -> The learning rate controls how much to change the weights according to the gradient of the loss function

        """
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def __mean_squared_error__(self,y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def __mean_squared_error_prime__(self,y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)
    
    def __binary_cross_entropy_prime__(self,y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    def __binary_cross_entropy__(self,y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Sayısal kararlılık için
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  

    def __mean_absolute_error_prime__(self,y_true, y_pred):
       
        grad = np.where(y_pred > y_true, 1, -1)
        return grad 

    def __mean_absolute_error__(self,y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))     

    def __error__(self,name:str,y_true,y_pred) ->float:
        """
        name can be {"mse" , "mae" , "bce"}
        """
        if name == "mse":
            return self.__mean_squared_error__(y_true,y_pred)
        elif name=="mae":
            return self.__mean_absolute_error__(y_true,y_pred)
        elif name=="bce":
            return self.__binary_cross_entropy__(y_true,y_pred)
        else:
            raise ValueError("Please check loss func name. İt can be 'mse','mae','bce' ")
        
    def __error_deriative__(self,name:str,y_true,y_pred) ->float:
        """
        name can be {"mse" , "mae" , "bce"}
        """
        if name == "mse":
            return self.__mean_squared_error_prime__(y_true,y_pred)
        elif name=="mae":
            return self.__mean_absolute_error_prime__(y_true,y_pred)
        elif name=="bce":
            return self.__binary_cross_entropy_prime__(y_true,y_pred)
        else:
            raise ValueError("Please check loss func name. İt can be 'mse','mae','bce' ")    


    def __binary_data_prep__(self, array) ->list:
        """
        This is written for models that are output with this sigmoid function.
        It is used to convert the predicted output into the desired output format.
        """

        if not np.any(array):
            return array  # if array is empty
        
        max_indeks = np.argmax(array)  # find the index of max
        
        # make zeros array, make max index 1 
        result = np.zeros_like(array)
        result[max_indeks] = 1

        return result

           

    def __write_list_to_file__(self,weights,biases,filename):
        """
        Saves weights and biases as a txt file to the specified file path.

        """
        for idx,i in enumerate(weights):
            weights[idx] = i.tolist()
        for idx,i in enumerate(biases):
            biases[idx] = i.tolist()
        

        if filename==None:
        # Şu anki tarih ve saati al
            now = datetime.now()
            # Dosya ismini tarih ve saat ile oluştur
            filename = now.strftime("%Y%m%d_%H%M%S.txt")
        
        # Dosyayı yazma modunda aç (dosya yoksa oluşturulur)
        with open(filename, 'w', encoding='utf-8') as file:
            # Her elemanı yeni bir satıra yaz
            for item in weights:
                file.write(f"{str(item)}\n")
            for item in biases:
                file.write(f"{str(item)}\n") 
        return filename          

    def __read_file_to_list__(self,filename)-> Union[list,list]:
        """
        Reads weights and biases from the specified file path and returns weights and biases as an array.
        """

        data_list = []
        # Dosyayı okuma modunda aç
        with open(filename, 'r', encoding='utf-8') as file:
            # Her satırı oku ve listeye ekle
            for line in file:
                # Satır sonundaki newline karakterini kaldır
                data_list.append(line.strip())

        mid = len(data_list)//2
        weights = data_list[:mid]
        biases = data_list[mid:]


        for idx,i in enumerate(weights):
            weights[idx] = np.array(ast.literal_eval(i))
        for idx,i in enumerate(biases):
            biases[idx] = np.array(ast.literal_eval(i))              
        return weights,biases

    

    def addActivations(self,function_name:str="sigmoid"):
        """
        function_name -> Name of the Activation function. it can be {"sigmoid" ,"tanh", "softmax", "relu"}
        Can be left blank. The sigmoid function is automatically assigned.
        
        """
        if len(self.layers)==0:
            raise ValueError("Please add a Dense or ConvolutionalLayer layer first")
        else:

            self.layers.append(ActivationLayer(function_name))

    def addConvolution(self,input_shape,kernel_size,kernel_count,stride:int=1):
        """
        input_shape -> shape of the input (with,height,channel)

        kernel_size -> size of the filter matrix e.x for 3 -> 3x3

        kernel count -> count of the filter matrix. 

        stride-> step count in convolution
        """
        flut = [layer for layer in self.layers if isinstance(layer, Flutten)]
        if len(flut)==0:
            self.layers.append(ConvolutionalLayer(input_shape,kernel_size,kernel_count,stride))
        else:
            raise ValueError("Cannot add a convolution layer after flattening")
    def flutten(self):
        Conv = [layer for layer in self.layers if isinstance(layer, ConvolutionalLayer)]
        pool = [layer for layer in self.layers if isinstance(layer, Pooling)]
        flut = [layer for layer in self.layers if isinstance(layer, Flutten)]
        if len(Conv) !=0: 

            if len(pool) ==0:
            

                if len(flut) ==0:
                    


                    
                    self.layers.append(Flutten(Conv[-1].output_shape,((Conv[-1].output_shape[0]*Conv[-1].output_shape[1]*Conv[-1].output_shape[2]),1)))
                else:
                    raise ValueError("You can only flutten once before the dense layer.")
            else:
                output_shape = (int((Conv[-1].out_with - pool[-1].poolsize) // pool[-1].stride + 1), int((Conv[-1].out_height- pool[-1].poolsize) // pool[-1].stride + 1),Conv[-1].out_channel)
                self.layers.append(Flutten(output_shape,(output_shape[0]*output_shape[1]*output_shape[2],1)))

        else:
            raise ValueError("Cannot flutten because model not have a Convolational layer")
        
    def addPooling(self,poolingtype,size:int=2,stride:int=2):
        self.layers.append(Pooling(poolingtype,size,stride))    
    def load_weights(self,filename:str):
        """
        filename -> File path where weights are saved

        Reads the weights and biases from the specified file path and assigns them to the model's weights and biases.

        """

        fclayers_idx = [index for index,layer in enumerate(self.layers) if isinstance(layer, DenseLayer)]

        weights,biases=self.__read_file_to_list__(filename)
        if len(fclayers_idx)!=len(weights):
            raise ValueError("Please make sure model or weights file are correct.")
        for idx,i in enumerate(fclayers_idx):
            self.layers[i].weights,self.layers[i].bias = weights[idx],biases[idx]
                        


    def summary(self):
        inp = [layer for layer in self.layers if isinstance(layer, ConvolutionalLayer)]
        if len(inp)!=0:
            x = np.random.random(inp[0].input_shape)
        else:
            inp = [layer for layer in self.layers if isinstance(layer, DenseLayer)]
            if len(inp)!=0:
                x = np.random.random((inp[0].weights[1],1)) 
            else:
                raise ValueError("Please verify the presence of convolution or dense layers in the model.")
        total = 0
        for i in self.layers:
            print("--------------------------------------------------------------------------------------------------")
            total += i.params
            x = i.forward(x)
            output_shape = "Output shape :" + str(x.shape)
            params = "             Trainable params :" + str(i.params)
            typee = (str(type(i))).replace("<class 'NeuralNetwork.","").replace(">","").replace("'","")
            empty = " "*77 + "|"
            
            print(f"{typee:<19} {empty}")
            empty = " "*13 + "|"
            print(f"{output_shape:<39} {params:<43} {empty} ")

            # print("Output shape :" + str(x.shape) + "             Trainable params :" + str(i.params))
            print("--------------------------------------------------------------------------------------------------")
        print("                                                     Total Trainable params :" + str(total))
            
        
    def plot(self,epoch,accuracy,loss):
        
        plt.figure(figsize=(10, 5))

        # Accuracy grafiği
        plt.plot(epoch, accuracy, label='Accuracy', marker='o', color='b')
        # Loss grafiği
        plt.plot(epoch, loss, label='Loss', marker='x', color='r')

        # Başlık ve etiketler
        plt.title('Model Accuracy and Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Value')

        # Efsane (legend)
        plt.legend()

        # Grafiği göster
        plt.grid(True)
        plt.show()


    def fit(self,x_train:list,y_train:list,loss_funtion:str="mse",epoch=20,learning_rate=0.001,accuarcy_type:str=None,filename:str=None):
        """
        x_train -> The lists of the training values. Shape must be (rows,columns,1)->(number of samples,number of inputs to neural network,1) 

        y_train -> The lists of the output. Shape must be (rows,columns,1)->(number of samples,number of outputs ,1) 

        loss_function -> Loss Function type for the training. İt can be {"mse","mae","bce"} 

        epoch -> Number of times the dataset will be trained

        learning_rate -> The learning rate controls how much to change the weights according to the gradient of the loss function

        accuarcy_type -> It is the parameter required for writing the success rate. The calculation procedures are different depending on the type of output.
        İt can be {"binary",}

        filename -> File path to save the weights. İt can be left blank. İf its blank current date and time will be name of the file.

        """
        loss_overall = []
        accuarcy_overall = []
        for ep in range(epoch):
            idx = 0
            err = 0
            error=0
            acc = 0
            for x,y in zip(x_train,y_train):
                
               
                output=self.forward(x)
                gradient = self.__error_deriative__(loss_funtion,y,output)
                self.backward(gradient,learning_rate)     
                err += self.__error__(loss_funtion,y,output)
                error = err/(idx+1)
                
                
                if accuarcy_type=="binary":
                    
                    predicted = self.__binary_data_prep__(output)
                    if np.array_equal(predicted,y):
                        acc +=1
                    os.system('cls')
                    accuarcy = acc/(idx+1)
                    print( "Sample : " + str(idx+1) +  " in " + str(len(x_train)) + "    Epoch : " + str(ep+1)  +" in "+str(epoch) + "    Loss : " + str(error)[0:9] + "    Accuarcy : " +str(accuarcy)[0:9])
                else:
                    os.system('cls')
                    print( "Sample : " + str(idx+1) +  " in " + str(len(x_train)) + "    Epoch : " + str(ep+1)  +" in "+str(epoch) + "    Loss : " +str(error)[0:9])
                idx +=1
            loss_overall.append(error)
            accuarcy_overall.append(accuarcy)
                
        weights=[]
        biases=[]

        fclayers = [layer for layer in self.layers if isinstance(layer, DenseLayer)]
        for layer in fclayers:
            weights.append(layer.weights) 
            biases.append(layer.bias)
            
        self.plot(range(epoch),accuarcy_overall,loss_overall)
        

        filename =self.__write_list_to_file__(weights,biases,filename)    
        print("Training Completed. Weights and Biases Saved at : " + str(filename))                 


    def makepics(self,directory,matrix):
        pass