
''' 
the main script to test deep neural network for different parameters. 
'''

from __future__ import print_function
import numpy as np

from tensorflow.python.client.session import Session

np.random.seed(1337)  # for reproducibility
import tensorflow as tf
import keras
from keras.utils import plot_model
#from keras.datasets import mnist
#from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback, History, EarlyStopping

from scipy.io import wavfile
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ownObjectives
import functions
import recoverSignal


#define variables -----------------------#
batch_size = 128
nb_classes = 10
nb_epoch = 30
windowLength = 256        
N_train = 15000000
N_test = 1050000
log_filepath = '/7.21_zhoukai'
 #------------------------ load data   ---------------------------# 
 
FsTrain, training_data = wavfile.read("7_21_2000originalTrainingFiles.wav")
 
 # for simplicity, the validation data is just the latter part of the training data
test_data = training_data[15000001:]
 
 #--------------- process data before neural network  ------------# 
 
dB_train, phase_train = functions.hanningFFT(training_data, N_train, windowLength)
dB_test, phase_test = functions.hanningFFT(test_data, N_test, windowLength)
 # all these are numpy.ndarray
 # stack 5 batches pr input row. 
 # each input row to the model is 645(129cheng5) samples if the windowLength is 256
X_train = functions.stack(dB_train, windowLength)
X_test = functions.stack(dB_test, windowLength)
 
 #--------------- define input and output of model ---------------# 
 
 # output of model is 129 samples. this is the true output value
Y_train = dB_train
Y_test = dB_test
 
 #-------------------------- make model --------------------------# 
inputShape = int((windowLength/2+1)*5)
outputShape = int(windowLength/2+1)
 
 # this returns a tensor
inputs =Input(shape=(inputShape,))
 
 # a layer instance is callable on a tensor, and returns a tensor
 # using keras can speed the building of the Model
x = Dense(1024, activation='sigmoid')(inputs)
x = Dropout(0.1)(x)
x = Dense(1024, activation='sigmoid')(x)
x = Dropout(0.1)(x)
predictions = Dense(outputShape, activation='linear')(x)
 
 # create model
model = Model(input=inputs, output=predictions)
model.summary()
 
 # --------------- define callback in order to be able to -----------# 
 # ------------- get the output from the model (test data)-----------# 
 #class classLossHistory(Callback):
 
    # def idefon_train_begin(self, logs={}):#exporting the .wav model outputting
   #      self.losses = []
    # def defon_batch_end(self, batch, logs={}):
   #      self.losses.append(logs.get('loss'))
     
 #po = classLossHistory()

 # ------------------- compile and train the model ------------------#
rms = RMSprop()
model.compile(loss=ownObjectives.lossFunction, optimizer=rms, metrics=["accuracy"])
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
cbks = [tb_cb]
 #sess.run()
# class printOutput(Callback):
# 
#     def getOutput(self):#exporting the .wav model outputting
#         get_output = K.function([model.layers[0].input], [model.layers[4].output])
#         layer_output = get_output([X_test])[0]
#         self.outputs = layer_output
#     
# po = printOutput()
 #early_stopping = EarlyStopping(monitor='val_loss', patience=1)
 # model.fit returns a history object which can be looked at 
h=model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch, verbose=2,
            validation_data=(X_test, Y_test),callbacks=cbks)

json_string=model.to_json()
model.save('7.21_Zhou_model')
model.save_weights('7.21_zhou_weights')
plot_model(model, to_file='7.21_zhou_model.png',show_shapes=True)



score = model.evaluate(X_test, Y_test, verbose=0)
predict=model.predict(X_test)
print('Test loss:', score[0])  
print('Test accuracy;', score[1])




#tf.saved_model


#print("output is updated")
 # the output is the result when the TEST signal is sent through the model. 
#plt.plot()
 # --------------------- restore signal and write file -------------------------#

true_output = recoverSignal.do_ifft(Y_test, phase_test)
output_model = recoverSignal.do_ifft(predict, phase_test)

int_true = true_output.astype("int16")
int_model = output_model.astype("int16")

wavfile.write('7.21_zhoukai_modeloutput.wav', 16000, int_model)
wavfile.write('7.21zhoukaitrue.wav', 16000, int_true)
print("files written")


