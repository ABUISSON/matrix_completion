import csv
import numpy as np
import random

def load_data():

    X_train = np.zeros((10000,4713))
    with open('input_train.csv', 'rt') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ')#, quotechar='|')
      for i,row in enumerate(spamreader):
          if i == 0:
              pass
          else:
              tar = row[0].split(';')[1]
              idx = tar.split('/')
              idx = [int(elem) for elem in idx]
              X_train[i-1,np.array(idx)] = 1


    X_test = np.zeros((10000,4713))
    with open('input_test.csv', 'rt') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ')#, quotechar='|')
      for i,row in enumerate(spamreader):
          if i == 0:
              pass
          else:
              tar = row[0].split(';')[1]
              idx = tar.split('/')
              idx = [int(elem) for elem in idx]
              X_test[i-1,np.array(idx)] = 1


    Y_train = np.zeros((10000,4713))
    with open('output_train.csv', 'rt') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ')#, quotechar='|')
      for i,row in enumerate(spamreader):
          if i == 0:
              pass
          else:
              tar = row[0].split(';')[1]
              idx = tar.split('/')
              idx = [int(elem) for elem in idx]
              Y_train[i-1,np.array(idx)] = 1

    permutation = np.random.permutation(X_train.shape[0])
    return X_train[permutation, :], Y_train[permutation, :], X_test


def get_training_matrices():
    '''
    X : training matrix
    W : mask : 1 if known, 0 if unknown
    Y_test : target matrix
    '''
    X_train, Y_train, X_test = load_data()
    X = X_train.copy()
    W = np.zeros(X.shape)
    masked = np.sort(random.sample(range(X_train.shape[0]),int(X_train.shape[0]/2)))
    Y_test = Y_train[masked,:]
    for i in range(X_train.shape[0]):
        if i in masked:
            for j in range(X_train.shape[1]):
                if X_train[i,j] == 1:
                    W[i,j] = 1
        else:
            X[i,:] = X_train[i,:] + Y_train[i,:]
            W[i,:] = np.ones(W.shape[1])

    return X, W, Y_test

def get_testing_matrices():
    '''
    X : testing matrix
    W : mask : 1 if known, 0 if unknown
    '''
    X_train, Y_train, X_test = load_data()
    X = np.zeros(((X_train.shape[0] + X_test.shape[0]), X_train.shape[1]))
    W = np.zeros(X.shape)
    X[:10000,:] = X_train + Y_train

    W[:10000,:] = np.ones(X_train.shape)
    W[10000:,:] = X_test

    permut = np.random.permutation(20000)
    return X[permut,:], W[permut,:], permut

def pred_from_mat(Y_pred, W):
    X_tar = np.multiply(Y_pred, 1-W)
    pred= []
    for i in range(X_tar.shape[0]):
        if i%1000 == 0:
            print('iter n : ', str(i))
        if np.sum(W[i,:]) != W.shape[1]:
            pred+=[list(X_tar[i,:].argsort()[-5:][::-1])]
    return pred


def scoring(pred_list, X_test, permut = None):
    '''
    The scoring function on the training set, several requirements:
    - pred_list: is a list of list (be carefull indices are aligned with X_test but not with real dataset (+1))
    5 elements inside each sub_list
    - X_test: reference dataframe, we observe if the predictions are in the set
    '''
    if permut:
        X = X[np.argsort(permut),:]
        W = W[np.argsort(permut),:]

    elem_count = 0
    prec = 0
    for i,l in enumerate(pred_list):
        for elem in l:
            elem_count += 1
            if int(X_test[i,elem]) == 1:
                prec += 1
    return prec/elem_count

def output_function(l_res, name = 'result.csv', mode = 'test'):
    with open(name,'w') as f:
        f.write('user_id' +';' +'items'+ '\n')
        for i in range(len(l_res)):
            if mode == 'train':
                f.write(str(i+1) + ';' + '/'.join(str(e+1) for e in l_res[i]) + '\n')
            else:
                f.write(str(i+10001) + ';' + '/'.join(str(e+1) for e in l_res[i]) + '\n')
    f.close()
