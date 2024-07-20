import pandas as pd
import numpy as np



def create_array(num):
    # Başlangıçta sıfırlar içeren bir NumPy dizisi oluştur
    arr = np.zeros(10, dtype=int)
    # Verilen numarayı dizi içinde ilgili konuma (index'e) ata
    if num < len(arr):
        arr[num] = 1
    return arr



def prepareData(filename):
    
    dataset = pd.read_csv(filename)
    x_train , y_train = dataset.iloc[:,1:],dataset.iloc[:,0]
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    y_train_normalized = np.array([create_array(num) for num in y_train])

    

    return x_train,y_train_normalized


def prepareData2(filename):
    # CSV dosyasını oku
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=1)
    
    # Verileri ayır
    y_train = dataset[:, 0]
    x_train = dataset[:, 1:]
    y_train = y_train.astype(int)
    # Verileri NumPy dizilerine dönüştür
    y_train_normalized = np.array([create_array(num) for num in y_train])
    
    return x_train, y_train_normalized