import numpy as np

def accuarcy(model,x_test,y_test):

    predicted = []
    for i in x_test:
        predicted.append(model.forward(i))
        
    predicted = np.array(predicted)
   
    predicted = predicted.reshape(predicted.shape[0],predicted.shape[1])

    binary_pred = []
    for i in predicted:
        binary_pred.append(model.__binary_data_prep__(i))

    y_test = y_test.reshape(y_test.shape[0],y_test.shape[1])
    binary_pred = np.reshape(binary_pred,(predicted.shape[0],predicted.shape[1],1))
    
    acc = 0
    for idx,pred in enumerate(binary_pred):
        
        if np.array_equal(pred.flatten(),y_test[idx]):
            acc +=1

    return acc,len(binary_pred),(acc/len(binary_pred))
        