from NeuralNetwork import Nn,Pooling

import numpy as np
from cal_accuarcy import accuarcy
from data_prep import prepareData


# train values
x_train,y_train = prepareData("C:\\Users\\merdg\\Desktop\\AI\\ML\\mnist_train.csv")
x_train,y_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) , y_train.reshape(y_train.shape[0],y_train.shape[1],1)


# test values
x_test,y_test = prepareData("C:\\Users\\merdg\\Desktop\\AI\\ML\\mnist_test.csv")
x_test,y_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1) , y_test.reshape(y_test.shape[0],y_test.shape[1],1)



model = Nn()
model.init_input_count(784)
model.addDense(32)
model.addActivations("relu")
model.addDense(10)
model.addActivations()

model.fit(x_train[0:10000],y_train[0:10000],"mse",10,0.001,"binary","model_mnist.qu")
model.load_weights("model_mnist.qu")
tre,total,acc = accuarcy(model,x_test,y_test)
print("True : " + str(tre) + " Total : " + str(total) + " Accuarcy : " + str(acc))

