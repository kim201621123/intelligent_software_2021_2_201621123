from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
import datetime as dt
import os
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

INIT_LR = 1e-4 
EPOCHS = 100
BS = 2


classes = "노랑나비속", "배추흰나비속", "호랑나비속", "not"

model_name = 'butterfly_e'+str(EPOCHS)+'_1213_9'

#dataset =  "./Cutted_Training_Image"
dataset = r"C:\Users\pinoc\Desktop\cap_new\tensorflow_mobilenetv2\images"
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
 # extract the class label from the filename
 label = imagePath.split(os.path.sep)[-2]
 # load the input image (224x224) and preprocess it
 image = load_img(imagePath, target_size=(224, 224))
 image = img_to_array(image)
 image = preprocess_input(image)
 # update the data and labels lists, respectively
 data.append(image)
 labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = preprocessing.LabelEncoder()
labels = lb.fit_transform(labels)

labels = to_categorical(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


# 모델 구조 정의 
model = Sequential()
model.add(Dense(32, input_shape=(224, 224, 3)))
model.add(Conv2D(32, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(7, 7)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (9, 9),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(7, 7)))

model.add(Flatten())    
model.add(Dense(256))   
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4))# 출력
model.add(Activation('softmax'))

# 모델 구축하기
"""
model.compile(optimizer=tf.keras.optimizers.Adam(),
 loss='categorical_crossentropy',
 metrics=['accuracy'])
"""

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer=opt,
    metrics=['accuracy'])

aug = ImageDataGenerator(
 rotation_range=20,
 zoom_range=0.15,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.15,
 horizontal_flip=True,
 fill_mode="nearest")


# Windows에서는 파일 이름에 / 대신 \\ 사용
dir = os.path.join('.\\log\\{}'.format(
 dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

callbacks = [
 # Write TensorBoard logs to `./logs` directory
 keras.callbacks.TensorBoard(log_dir= dir, profile_batch = 0)
]

H = model.fit(
 aug.flow(trainX, trainY, batch_size=BS),
 steps_per_epoch=len(trainX) // BS,
 validation_data=(testX, testY),
 validation_steps=len(testX) // BS,
 epochs=EPOCHS)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
 target_names=lb.classes_))

print("[INFO] saving butterfly model...")
model.save(model_name+'.h5')
summary = model.summary()

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(model_name+'.png', dpi=300)


