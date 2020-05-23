import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras import applications



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



model = applications.MobileNet(include_top=True, weights=None, input_shape=(64,64,3),classes=classes)
model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',metrics=['accuracy'])
model.summary()



#Set checkpoint
checkpoint = ModelCheckpoint(filepath='Result/Mo_64_ckpt.hdf5',monitor='val_acc',
                             verbose=1,save_best_only=True)
def lr_sch(epoch):
    if epoch <20:
        return 1e-2
    if 20<epoch <40:
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
history = model.fit(x_train,y_train,batch_size=64,epochs=100,validation_split=0.3,validation_data=None,verbose=1,callbacks=None)

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
plt.savefig('Result_plot/Mo_64_confusion.png')
plt.show()


#Save history
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'Save_Data/history_Mo_64.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)




