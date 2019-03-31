

# Create by: Joelson Antônio dos Santos in January-02-2019.
# kNN classifier using brute force search (naive approach) for predict label for observations (not seen points)

# utils
import numpy as np
import pandas as pd
import collections
import sys

# metrics
from scipy.spatial import distance

# data split approach
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris # just for tests

class KNN:
    
    def __init__(self, k, distanceFunction):
        self.setK(k)
        self.setDistanceFunction(distanceFunction)
        self.setLabels([])
        
    def setK(self, k):
        self.k = k
        
    def setDistanceFunction(self, distanceFunction):
        self.distanceFunction = distanceFunction
        
    def setLabels(self, labels):
        self.labels = labels
        
    def getK(self):
        return self.k
    
    def getDistanceFunction(self):
        return self.distanceFunction
    
    def getLabels(self):
        return self.labels
    
    # some metric (dis-similarities) provided by scipy library (package)
    def metric(self, p1, p2):
        if self.distanceFunction == "euclidean":
            return distance.euclidean(p1, p2)
        elif self.distanceFunction == "manhattan":
            return distance.cityblock(p1, p2)
        elif self.distanceFunction == "cosine":
            return distance.cosine(p1, p2)
        else:
            return distance.euclidean(p1, p2) # default
        
    def predict(self, X_train, X_test, y_train):
        row_train = len(X_train) - 1
        row_test = len(X_test) - 1
        self.setLabels(np.zeros(len(X_test)))
        for point in range(0, row_test): 
            if self.k == 1:
                kNNDistance = sys.float_info.max
            else:
                kNNDistances = np.zeros(self.k)
                kNNeighbor = np.zeros(self.k)
                for i in range(0, self.k - 1):
                    kNNDistances[i] = sys.float_info.max
            for neighbor in range(0, row_train):
                # compute distance
                distance = self.metric(X_test[point,], X_train[neighbor,])
                # print("object: ", neighbor, " distance: ", distance, y_train[neighbor])
                if self.k == 1: # one-nn
                    if kNNDistance > distance:
                        kNNDistance = distance
                        self.labels[point] = y_train[neighbor]
                else:
                    shiftLeft = len(kNNDistances) - 1
                    while shiftLeft > 0 and kNNDistances[shiftLeft] > distance:
                        shiftLeft = shiftLeft - 1
                    if shiftLeft == 0:
                        if kNNDistances[shiftLeft] > distance:
                            for i in range(len(kNNDistances) - 1, 0, -1):
                                kNNDistances[i] = kNNDistances[i - 1]
                                kNNeighbor[i] = kNNeighbor[i - 1]
                            kNNDistances[0] = distance
                            kNNeighbor[0] = y_train[neighbor]
                        else:
                            for i in range(len(kNNDistances) - 1, 1, -1):
                                kNNDistances[i] = kNNDistances[i - 1]
                                kNNeighbor[i] = kNNeighbor[i - 1]
                            kNNDistances[1] = distance
                            kNNeighbor[1] = y_train[neighbor]
                    else:
                        for i in range(len(kNNDistances) - 1, shiftLeft, -1):
                            kNNDistances[i] = kNNDistances[i - 1]
                        kNNDistances[shiftLeft] = distance
                        kNNeighbor[shiftLeft] = y_train[neighbor]
            if self.k > 1:
                # count freq.
                counter = collections.Counter(kNNeighbor)
                maximum = 0
                predicted_label = -1
                for key, value in counter.items():
                    if maximum < value:
                        maximum = value
                        predicted_label = key
                self.labels[point] = predicted_label
                    
# test method (can pass dataset[array numpy] or file name as argument)           
def runner(k, path_file, distanceFunction):
    # reading a dataset from file
    #dataset = np.loadtxt(path_file, dtype='float', delimiter=',')
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)
    model = KNN(k, distanceFunction)
    model.predict(X_train, X_test, y_train)
    # label clusters
    print("kNN algorithm using", model.getDistanceFunction(), "distance: (score) -> {:.2f} ".format(np.mean(model.getLabels() == y_test)))

