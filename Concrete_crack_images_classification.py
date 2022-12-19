
#%%
#import module
import os
import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras, data
from keras import Input, optimizers, losses, callbacks, applications
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation 
from keras.utils import image_dataset_from_directory, plot_model
from keras. callbacks import TensorBoard, EarlyStopping




#%%
#Load Dataset
filepath = os.path.join(os.getcwd(), 'Dataset')
BATCH_SIZE = 32
IMG_SIZE = (160,160)

train_ds = image_dataset_from_directory(
  filepath,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)


val_ds = image_dataset_from_directory(
  filepath,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(class_names)




#%%
#Data Inspection

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")



#%%
# Data Preprocessing 
#Convert BatchDataset into PrefetchDataset
AUTOTUNE = data.AUTOTUNE
pf_train = train_ds.prefetch(buffer_size=AUTOTUNE)
pf_test = val_ds.prefetch(buffer_size=AUTOTUNE)

#Image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(RandomFlip('horizontal'))
data_augmentation.add(RandomRotation(0.2))

#Apply data augmentation on one image and see the result
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')



# %%
# Create the layer for data normalization
preprocess_input = applications.mobilenet_v2.preprocess_input

#Apply transfer learning to create the feature extractor
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
base_model.trainable = False



# %%
#Create classifier
global_avg = GlobalAveragePooling2D()
h1 = Dense(64,activation='relu')
output_layer = Dense(len(class_names),activation='softmax')

# Link all the parts together with functional API
inputs = Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)

s = Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#Model Architecture
plot_model(model, show_shapes=True)




# %%
#Compile model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'])


#Callbacks
logdir = os.path.join(os.getcwd(), 
    'logs', 
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

es = EarlyStopping(monitor='val_loss',
    patience=2)

tb = TensorBoard(log_dir=logdir)

#Train Model
history = model.fit(
    pf_train,
    validation_data=pf_test,
    epochs=20,
    callbacks=[es,tb])



# %%
#Model deployment
#Use the model to perform prediction
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

print("Classification Report:\n----------------------\n", 
    classification_report(label_batch, y_pred)
    )

#Stack label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))


#%%

