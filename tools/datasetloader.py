import numpy as np
import cv2
import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("processed {}/{}".format(i + 1,
                    len(imagePaths)))
        

        tf.keras.backend.set_image_data_format('channels_last')
        temp_x_train = []
        for i in range(len(data)):
            new_x_train_row=np.moveaxis(data[i],0,2)
            temp_x_train.append(new_x_train_row)
        data = np.array(temp_x_train)

        # print("SimpleDatasetLoader input shape2 data: ", np.array(data).shape)

        # print("SimpleDatasetLoaderinput shape1 label: ", np.array(labels).shape)

        tf.keras.backend.set_image_data_format('channels_last')
        temp_x_train = []
        for i in range(len(labels)):
            new_x_train_row=np.moveaxis(labels[i],0,2)
            temp_x_train.append(new_x_train_row)
        labels = np.array(temp_x_train)

        # print("SimpleDatasetLoader input shape2 label: ", np.array(labels).shape)

        return (np.array(data), np.array(labels))

class NiiDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, data_folder, modalities):

        data = []
        labels = []
        for folder in sorted(glob.glob(os.path.join(data_folder, "*"))):
            print('Processing foder: ' + folder)
            img = []
            for modality in modalities:
                filename = glob.glob(os.path.join(folder, '*/*%s.*.nii' % modality))[0]
                image = nib.load(filename).get_fdata()
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                img.append(image)
            data.append(np.array(img))
            filename = glob.glob(os.path.join(folder, '*/*OT*.nii'))
            # print(filename)
            if filename == []:
                label = None
            else:
                label= nib.load(filename[0]).get_fdata()
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        label = p.preprocess(label)
                label = np.expand_dims(label, axis=0)
            labels.append(np.array(label))

        # print("NiiDatasetLoader input shape1 data: ", np.array(data).shape)

        tf.keras.backend.set_image_data_format('channels_last')
        temp_x_train = []
        for i in range(len(data)):
            new_x_train_row=np.moveaxis(data[i],0,2)
            temp_x_train.append(new_x_train_row)
        data = np.array(temp_x_train)

        # print("NiiDatasetLoader input shape2 data: ", np.array(data).shape)

        # print("NiiDatasetLoader input shape1 label: ", np.array(labels).shape)

        tf.keras.backend.set_image_data_format('channels_last')
        temp_x_train = []
        for i in range(len(labels)):
            new_x_train_row=np.moveaxis(labels[i],0,2)
            temp_x_train.append(new_x_train_row)
        labels = np.array(temp_x_train)

        # print("NiiDatasetLoader input shape2 label: ", np.array(labels).shape)

        return (data, labels)

class NiiImagesLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, folder, modalities):
        print('Processing foder: ' + folder)
        data = []
        images = [] 
        for modality in modalities:
            filename = glob.glob(os.path.join(folder, '*/*%s.*.nii' % modality))
            print("filename: ", filename)
            filename = filename[0]
            img = nib.load(filename).get_fdata()
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    img = p.preprocess(img)
            data.append(img)

        # print("NiiImagesLoader input shape1: ", np.array(data).shape)

        tf.keras.backend.set_image_data_format('channels_last')
        temp_x_train = []
        for i in range(len(data)):
            new_x_train_row=np.moveaxis(data[i],0,2)
            temp_x_train.append(new_x_train_row)
        data = np.array(temp_x_train)

        # print("NiiImagesLoader input shape2:  ", np.array(data).shape)

        filename = glob.glob(os.path.join(folder, '*/*OT*.nii'))
        if filename == []:
            label = None
        else:
            label= nib.load(filename[0]).get_fdata()
            label= np.moveaxis(label,0,2)
        return (np.array(data), label)
