# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:18:53 2019

@author: BHUSHAN
"""
import os
os.chdir('F:\\Python\\AV\\apparel')
os.getcwd()

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plot

from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.pyplot import imread
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing train data (60,000 images) set 
folder='train'
#sort the image name for proper importing in sequence
order_img_name=sorted(os.listdir(folder), key=lambda x: int(x.partition('.')[0]))
TR_X, TS_X = [],[] #define 2 list for train & test
for img_name in order_img_name :     
    img = plot.imread('train/' + img_name)
    img=img[:,:,0] # slice the 4 channel to 0
    TR_X.append(img)
TR_X = np.array(TR_X)
TR_X1 = TR_X.reshape((TR_X.shape[0], 28, 28, 1)) #reshape to 4 dimesniosn
#print sample images
plot.imshow(TR_X[9].reshape(28,28))

#importing test data (10,000 images) set
folder_test='test'
#sort the image name for proper importing in sequence
sorted_test_img=sorted(os.listdir(folder_test), key=lambda x: int(x.partition('.')[0]))
for img_name in sorted_test_img : 
    img = plot.imread('test/' + img_name)
    img=img[:,:,0] # slice the 4 channel to 0
    TS_X.append(img)
TS_X = np.array(TS_X)
TS_X1 = TS_X.reshape((TS_X.shape[0], 28, 28, 1))#reshape to 4 dimesniosn

# convert from integers to floats
#TR_X1 = TR_X.astype('float32')
#TS_X1 = TS_X.astype('float32')
#import y files and preprocess to match
TR_Y = pd.read_csv('train.csv')
TS_Y = pd.read_csv('test.csv')
TR_Y.head(3)
TR_Y=TR_Y.drop(columns="id")
TR_Y=np.array(TR_Y ,dtype='f')
TR_Y1 = to_categorical(np.array(TR_Y))

#load data fashion-mnist pre-shuffled train data and test data for testing our solution
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
testY1 = to_categorical(testY)

#define CNN model fifth training model used # NEXT USE dropout 0.3
def define_model():
    # first CONV => RELU => CONV => RELU => POOL layer set
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu', name="conv1",kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=1,name="batch_nomr1"))
    model.add(Conv2D(32, (3, 3),padding="same",  activation='relu',name="conv2",kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=1,name="batch_norm2"))
    model.add(MaxPooling2D((2, 2),name="max_pool1")) #using average pooling
    model.add(Dropout(0.25, name="drop1"))
    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same",activation='relu', name="conv3",kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=1,name="batch_norm3"))
    model.add(Conv2D(64, (3, 3), padding="same",activation='relu', name="conv4",kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=1,name="batch_norm4"))
    model.add(MaxPooling2D(pool_size=(2, 2),name="max_pool2"))#using average pooling
    model.add(Dropout(0.25, name="drop2"))
    # third CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same",activation='relu', name="conv5",kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=1,name="batch_norm5"))
    model.add(Conv2D(128, (3, 3), padding="same",activation='relu', name="conv6",kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=1,name="batch_norm6"))
    model.add(Dropout(0.25, name="drop3"))
    # first (and only) set of FC => RELU layers
    model.add(Flatten(name="flat1"))
    model.add(Dense(768, activation="relu",kernel_initializer='he_uniform', name="dense1"))
    model.add(Dropout(0.3,name="drop4"))#
    model.add(Dense(512, activation="relu",kernel_initializer='he_uniform', name="dense2"))
    model.add(Dropout(0.5,name="drop5"))#
    # softmax classifier
    model.add(Dense(10,activation="softmax", name="output"))
    # compile model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model=define_model()
print(model.summary())

import datetime
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = StratifiedKFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX,dataY): # kfold.split: Generate indices to split data into training and test set.
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		trainY = to_categorical(trainY)
		testY = to_categorical(testY)
		# fit model
		start=datetime.datetime.now()
		print(start) # add to training callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True)]
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
		mid=datetime.datetime.now()
		print(mid)
		# evaluate model
		print("Saved model to disk")
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))# 92.617; 92.742; 
		model.save('apparel_model.h5')
		model.save_weights('fashion-mnist-weights.h5') #to be run tih saving weights
		# list all data in history
		#print(history.history.keys())
		print(model.metrics_names)
		#print("acc:%d loss:%d val_acc:%d val_loss:%d" % history["acc"],history["loss"], history["val_acc"], history["val_loss"])
		# append scores
		scores.append(acc)
		histories.append(history)
		end=datetime.datetime.now()
		print(end)
		mid-start
		end-mid
		end-start
	return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['acc'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()
    

# k-fold Cross validation for model stability
scores, histories = evaluate_model(TR_X1, TR_Y)
print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
# learning curves
summarize_diagnostics(histories)
# summarize estimated performance
summarize_performance(scores)


# train final model on full training dataset
history1 = model.fit(TR_X1, TR_Y1, epochs=30, batch_size=32, verbose=1)
# save the model & weights
model.save('apparel_fin_2_mode.h5')
model.save_weights('fashion_mnist_weights.h5') 
#prediction for test data set.
model = load_model('apparel_fin_2_mode.h5')
yhat1 = model.predict(TS_X1)# probablilitoes prediction
yhat11 = model.predict_classes(TS_X1)# class prediction
yhat11[:20]
yhat11_categ = to_categorical(np.array(yhat11))

accuracy_score(testY,yhat11) # accuracy on TEST data set 93.63%
#plot of loss & accuracy
def plot_loss_acc(history1):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history1.history['loss'], color='blue', label='train')
	#pyplot.plot(history1.history['val_loss'], color='orange', label='test')
	# plot accuracy
	#pyplot.subplot(212)
	#pyplot.title('Classification Accuracy')
	pyplot.plot(history1.history['accuracy'], color='blue', label='train')
	#pyplot.plot(history1.history['val_accuracy'], color='orange', label='test')
	plot.legend(loc="upper left")
	pyplot.show()
plot_loss_acc(history1)    
    
#load saved finalmodel trained on 60k training data with 30 epochs
model=load_model('apparel_fin_2_mode.h5') 

#run on test data set and check accuracy
model.evaluate(TS_X1, testY1, verbose=2) # accuracy on TEST data set 93.63%
#print classwise precision,recall & f1score on TEST data set.
labelNames=["Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"]
print(classification_report(testY, yhat11,target_names=labelNames))
#function plots a confusion matrix
def plot_confusion_matrix(cm,classes,title='Confusion matrix',cmap=plot.cm.Wistia):
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    #title='Confusion matrix'
    #plot.title(title)
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=90)
    plot.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="black" if cm[i, j] > thresh else "black")
    plot.ylabel('True labels')
    plot.xlabel('Predicted labels')
    plot.show()
plot_confusion_matrix(confusion_matrix(testY, yhat11), labelNames)

#submission file generation
submission = pd.concat([pd.Series(TS_Y.id,name = "id"),pd.Series(yhat11,name="label")],axis = 1)
yhat11[9981] #check sample
submission.to_csv("test_results.csv",index=False)

