import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K


#Test Gpu
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())
K.tensorflow_backend._get_available_gpus()


#Load data
x_train = np.load('../Data/train_data_64.npy')
y_train = np.load('../Data/train_lbl_64.npy')
x_test = np.load('../Data/test_data_64.npy')
y_test = np.load('../Data/test_lbl_64.npy')
classes = 5

#Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Sequential
model = Sequential()
model.add(Conv2D(input_shape=x_train.shape[1:],filters=16,kernel_size=2, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(5,activation='softmax'))


#model initilize


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



#Set checkpoint
checkpoint = ModelCheckpoint(filepath='Result/CNN3_64_ckpt.h5',monitor='val_acc',
                             verbose=1,save_best_only=True)
def lr_sch(epoch):
    if epoch <20:
        return 1e-2
    if 20<=epoch <40:
        return 1e-3
    if 40<=epoch<60:
        return 1e-4
    if 60<=epoch<80:
        return 1e-5  
    if epoch>=80:
        return 1e-6
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,
                               mode='max',min_lr=1e-3)
callbacks = [checkpoint,lr_scheduler,lr_reducer]


#Train
history = model.fit(x_train,y_train,batch_size=64,epochs=100,validation_split=0.3,validation_data=None,verbose=1,callbacks=callbacks)

#Test
scores = model.evaluate(x_test,y_test,verbose=1)
print('Test loss:',scores[0])
print('Test accuracy:',scores[1])

#Confustion matrix
y_pred=model.predict(x_test)
y_test = np.argmax(y_test,axis = 1)
y_pred = np.argmax(y_pred,axis = 1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm= cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
print(cm)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)
plt.savefig('Result_plot/CNN3_64_confusion.png')
plt.show()


#Save history
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'Result/history_CNN3_64.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)




