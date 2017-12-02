#import images from the given path
#from several directories
#Images will be loaded, normalized, shuffled
#resized, labeled, and flattened
#everything will be saved as .dat file
#using pickle in a dictionary of numpy arrays

##jump to the handler function as it provides
##all the parameters that you have to supply
##it will produce the .dat file

##note that all images are read in grayscale
##you can change that in the getImages lambda
#to cv2.IMREAD_BGR
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import argparse

normalize = lambda lst,channels:  tf.nn.l2_normalize(lst, dim=channels-1)
getImages = lambda pathF: [cv2.imread(os.path.join(pathF, i), cv2.IMREAD_GRAYSCALE) for i in os.listdir(pathF)]

def flatten(lst):
    tmp = []
    for i, j in enumerate(lst):
        tmp.append(j.flatten())
    return np.array(tmp)

#hot vector
#pass a list of sizes of each class
def hotVector(lst,numClasses):
    value = 0
    labelList = []
    for i in lst:
        labels = np.full(i, value)
        labelList.append((np.arange(numClasses) == labels[:,None]).astype(np.float32))
        value += 1    
    return np.concatenate((labelList), axis=0)

#labeling
#pass a list of sizes of each class
def labelIt(lst):
    value = 0
    labelList = []
    for i in lst:
        labelList += ([value] * i)
        value += 1
    return np.array(labelList)

def handler(dataPath='./Fnt', channels=1, numClasses=36, newSize=None,
            trainFraction=0.8, hotVec=False, shuffle=True, flat=True, normal=True):
    """
    Parameters:
    
    path: string, the path that contains all the files. Each file should be
    a class of its own

    channels: int, the channels of the image (3 for RGB, 1 for black&white)

    numClasses: int, the number of classes to be imported. This is equivalent
    to the number of files (each file contains one class of images). if
    numClasses<available directories than it will take the first numClasses listed

    newSize: tuple, the size to be changed (using cv2.resize)
    if None it won't be resized. Note that the images are preferred
    to be the same size (if you are not willing to resize).
    I didn't try what happens when not resizing, maybe nothing.

    trainFraction: float, some of the images will be taken for
    training and some for testing

    hotVec: boolean, ift rue the labels will be hot vectors
    otherwise regular labeled vector

    shuffle: boolean, if true will shuffle the data

    flat: boolean, if true will flatten the image

    normal: boolean, if true will normalize the image    
    """

    trainSet = []
    testSet = []
    trainLabels = []
    testLabels = []

    #extract for each class/directory
    count = 0
    for i in os.listdir(dataPath)[:numClasses]:
        tmp = getImages(os.path.join(dataPath, i))
        currentSet = []
        actualSize = tmp[0].shape
        #resize
        if newSize is not None:
            actualSize = newSize
            for j in tmp:
                currentSet.append(cv2.resize(j, newSize))
        else:
            currentSet = tmp
        
        trainingSize = int(len(currentSet)*trainFraction)
        testingSize = len(currentSet) - int(len(currentSet)*trainFraction)
        trainSet.append(currentSet[:trainingSize])
        testSet.append(currentSet[trainingSize:])
        trainLabels.append([count]*trainingSize)
        testLabels.append([count]*testingSize)
        count += 1
        
    #name of pickle file
    file = 'data'+str(numClasses)+'_classes_'+str(actualSize[0])+'x'+str(actualSize[1])
    if flat:
        file += '_flat'
    if normal:
        file += '_normalized'
    if hotVec:
        file += '_hotVector'
    file += '.dat'
        
    #concatenate
    trainSet = np.concatenate((trainSet), axis=0)
    testSet = np.concatenate((testSet), axis=0)
    trainLabels = np.concatenate((trainLabels), axis=0)
    testLabels = np.concatenate((testLabels), axis=0)

    #convert to normalize
    #converting data to float yields better results
    #even if you dont normalize
    trainSet = np.array(trainSet).astype(np.float32)
    testSet = np.array(testSet).astype(np.float32)

    #hot vector
    if hotVec:
        trainLabels = hotVector([trainingSize]*numClasses,numClasses)
        testLabels = hotVector([testingSize]*numClasses,numClasses)
    else:
        #label 
        trainLabels = labelIt([trainingSize]*numClasses)
        testLabels = labelIt([testingSize]*numClasses)

    #shuffle
    if shuffle:
        with tf.Session():
            trainSet = tf.random_shuffle(trainSet, 5).eval()
            testSet = tf.random_shuffle(testSet, 6).eval()
            trainLabels = tf.random_shuffle(trainLabels, 5).eval()
            testLabels = tf.random_shuffle(testLabels, 6).eval()

    #normalize  
    if normal:
        with tf.Session():
            trainSet = normalize(trainSet, channels).eval()
            testSet = normalize(testSet, channels).eval()

    #flatten
    if flat:
        trainSet = flatten(trainSet)
        testSet = flatten(testSet)

    #save   
    save = {
        'trainSet': trainSet,
        'testSet': testSet,
        'trainLabels': trainLabels,
        'testLabels': testLabels,
        'imgSize': actualSize,
        'normal': normal,
        'numClasses': numClasses,
        'shuffle': shuffle,
        'hotVec': hotVec,
        'channels': channels
        }
    with open(file, 'wb') as f:
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
