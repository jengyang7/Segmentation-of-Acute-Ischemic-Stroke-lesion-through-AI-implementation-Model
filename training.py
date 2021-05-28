import os
import tensorflow as tf
import numpy as np
import pickle
import math

from tools import BatchGenerator
from tensorflow.python.client import device_lib
from tools import NiiDatasetLoader
from tools import Resize
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger
from models import isensee2017


print(device_lib.list_local_devices())

# model settings
config["input_size"] = (128,128) # size of image (x,y)
config["nr_slices"] = (32,) # all inputs will be resliced to this number of slices (z axis), must be power of 2
config["modalities"] = ["CT"]   # we only train on the CT scan from the ISLES dataset since user only uoload CT scan
config["mean"] = [144.85851701] #   the obtained mean
config["std"] = [530.26314967]  # the obtained standard deviation
config["labels"] = 1 # the label numbers on the input image 

config = dict()
# directories
config["output_folder"] = os.path.abspath("gdrive/My Drive/code/output_5/") # path save the output
config["data_folder"] = os.path.abspath("gdrive/My Drive/code/TRAINING/") #path to folder containing the training data
config["tensorboar_log_dir"] = os.path.join(config["output_folder"], "log")
config["model_folder"] = os.path.join(config["output_folder"], "model")
if not os.path.exists(config["model_folder"]):
    os.makedirs(config["model_folder"])
config["model_file"] = os.path.join(config["model_folder"], "model.h5")
config["wieghts_file_bestval"] = os.path.join(config["model_folder"], "BEST_val.h5")
config["wieghts_file_lasttrain"] = os.path.join(config["model_folder"], "LAST_train.h5")
config["training_file"] = os.path.join(config["output_folder"], "training_ids.pkl")
config["validation_file"] = os.path.join(config["output_folder"], "validation_ids.pkl")
config["logging_file"] = os.path.join(config["output_folder"], "training.log")
config["overwrite"] = False  # If True, will overwite previous files


# training settings
config["batch_size"] = 4
config["n_epochs"] = 350  # training stopped at 350 epoch
config["initial_learning_rate"] = 5e-3
config["learning_rate_drop"] = 0.5  # decay the learning by 0.5 factor
config["test_size"] = 0.2  # portion of the training data for validation

# load the dataset from disk
preprocessor = [Resize(config["input_size"]+config["nr_slices"])]
(data, labels) = NiiDatasetLoader(preprocessor).load(config["data_folder"], config["modalities"])

#  if output folder exist, create it
if not os.path.exists(config["output_folder"]):
    os.makedirs(config["output_folder"])

# normalize data
def normalize_data(data, mean, std):
    for i in range(len(data)):
        for j in range(len(mean)):
            data[i][j] -= mean[j]
            data[i][j] /= std[j]
    return data 
data = normalize_data(data, config["mean"], config["std"])

# split the dataset into training and testing 
if not os.path.exists(config["validation_file"]) or config["overwrite"]:
    from sklearn.model_selection import train_test_split
    indices = [x for x in range(len(data))]
    training_indices, validation_indices = train_test_split(indices, test_size=config["test_size"])
    with open(config["validation_file"], "wb") as opened_file:
        pickle.dump(validation_indices, opened_file)
    with open(config["training_file"], "wb") as opened_file:
        pickle.dump(training_indices, opened_file) 
else:
    with open(config["validation_file"], "rb") as opened_file:
        validation_indices = pickle.load(opened_file)
    with open(config["training_file"], "rb") as opened_file:
        training_indices = pickle.load(opened_file)

trainX = [data[i] for i in training_indices]
trainY = [labels[i] for i in training_indices]
testX = [data[i] for i in validation_indices]
testY = [labels[i] for i in validation_indices]

trainX = tf.stack(trainX)
trainY = tf.stack(trainY)
testX = tf.stack(testX)
testY = tf.stack(testY)

# create or load model
if os.path.exists(config["model_file"]):
    from utils import ModelLoader
    model = ModelLoader(config["model_file"])

else:
    input_shape=(len(config["modalities"]),)+config["input_size"]+config["nr_slices"]
    model = isensee2017(input_shape=input_shape,
                              n_labels=config["labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              optimizer="Adam")

print("Model compiled")

# decay learning rate
def poly_step_decay(epoch):

    initLR = config["initial_learning_rate"]
    endLR = config["initial_learning_rate"]*1e-7
    dropEvery = config["n_epochs"]
    power = 0.9

    # compute learning rate for the current epoch
    dropEvery = dropEvery * math.ceil((epoch+1) / dropEvery)
    LR = (initLR - endLR) * (1 - epoch / dropEvery) ** (power) + endLR
   
    return float(LR)

# create training callbacks
callbacks = [LearningRateScheduler(poly_step_decay, verbose=1),
             ModelCheckpoint(monitor='val_loss',
                             filepath=config["model_file"],
                             save_best_only=True,
                             save_weights_only=False),
             ModelCheckpoint(monitor='val_loss',
                             filepath=config["wieghts_file_bestval"],
                             save_best_only=True,
                             save_weights_only=True),
             ModelCheckpoint(monitor='val_loss',
                             filepath=config["wieghts_file_lasttrain"],
                             save_best_only=False,
                             save_weights_only=True),
             TensorBoard(log_dir=config["tensorboar_log_dir"]),
             CSVLogger(config["logging_file"], append=True)]


model.fit(x=trainX, y=trainY, batch_size=config["batch_size"],
                    steps_per_epoch=(float(len(trainX)) / config["batch_size"]),
                    epochs=config["n_epochs"],
                    
                    callbacks=callbacks,
                    validation_data=(testX, testY),
                    validation_steps=(float(len(testX)) / config["batch_size"]),
                    verbose=1
                    )

model.summary()