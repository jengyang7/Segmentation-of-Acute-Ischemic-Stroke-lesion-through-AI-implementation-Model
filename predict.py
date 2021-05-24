import os
import nibabel as nib
import numpy as np
import pickle
import glob
import argparse

# create Argument parser
parser = argparse.ArgumentParser(
    description='Prediction')

parser.add_argument("--i", default="input", type=str,
                    help='Input image and xml directory')

parser.add_argument("--o", default="output", type=str,
                    help='Output augmeneted image and xml directory.') 

args = parser.parse_args()

config = dict()
config["output_folder"] = os.path.abspath("output/")  # outputs path
config["training_file"] = os.path.join(config["output_folder"], "training_ids.pkl")
config["validation_file"] = os.path.join(config["output_folder"], "validation_ids.pkl")
config["model_folder"] = os.path.abspath("model/")
if not os.path.exists(config["model_folder"]):
    os.makedirs(config["model_folder"])
config["model_file"] = os.path.join(config["model_folder"], "model.h5")
config["weights_file_bestval"] = os.path.join(config["model_folder"], "BEST_val.h5")
config["weights_file_lasttrain"] = os.path.join(config["model_folder"], "LAST_train.h5")
config["weights_file"] = config["weights_file_lasttrain"]
config["prediction_path"] = os.path.join(config["output_folder"], "prediction/")

# model settings
config["input_size"] = (128,128) # the size of image 
config["nr_slices"] = (32,) # all inputs will be resliced to this number of slices (z axis), must be power of 2
config["label"] = (1,) 
config["n_base_filters"] = 16
config["threshhold"] = 0.1 # threshold used to convert output heat map to output mask with 0 and 1 only, i.e. >thresh => 1
config["mean"] = [144.85851701] # found mean from training data
config["std"] = [530.26314967]  # # found sd from training data


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    return np.mean(2. * (np.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(np.sum(y_true,
                                                            axis=axis) + np.sum(y_pred,
                                                                               axis=axis) + smooth))
                             
import numpy as np
import cv2
import os
import glob
import nibabel as nib
import numpy as np


### Resize ###

import shutil
import numpy as np
import SimpleITK as sitk

def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2

def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image

def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkLinear, transform=None,
                           output_pixel_type=None):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)

def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)

def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0, 2))
    return data

def data_to_sitk_image(data, spacing=(1., 1., 1.)):
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=np.float))
    return image

def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.):
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation is "linear":
        interpolator = sitk.sitkLinear
    elif interpolation is "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError("'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(
            interpolation))
    resampled_image = sitk_resample_to_spacing(image, new_spacing=target_spacing, interpolator=interpolator,
                                               default_value=default_value)
    return sitk_image_to_data(resampled_image)

class Resize:
    def __init__(self, new_shape,  interpolation="linear"):
        self.new_shape = new_shape  # new shape of output image
        self.interpolation = interpolation  #either "linear" ot "nearest"
    def preprocess(self, image,):   #image : (np.array, 3 dims: x,y,z)
        zoom_level = np.divide(self.new_shape, image.shape)
        new_spacing = np.divide((1.,1.,1.), zoom_level)
        new_data = resample_to_spacing(image, (1.,1.,1.), new_spacing, interpolation=self.interpolation)
        return new_data

### ModelLoader ###

from keras.models import load_model

from keras import backend as K

def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))

def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1-weighted_dice_coefficient(y_true, y_pred)

def ModelLoader(model_file):
    print("Loading pre-trained model")
    custom_objects = {'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


import nibabel

def predict(filename, mean, std, model, output_filename):
    # load data
    preprocessors = [Resize(config["input_size"]+config["nr_slices"])] # prepare preprocessor for resizing
    # if the preprocessors are None, initialize them as an empty list
    if preprocessors is None:
        preprocessors = []
    # initialize the list of features and labels
    data = []
    images = [] 
    print("filename: ", filename)
    img = nib.load(filename).get_fdata()
    # check if preprocessors are not None
    if preprocessors is not None:
        # loop over the preprocessors and apply to each image
        for p in preprocessors:
            img = p.preprocess(img)
    data.append(img)
    # convert data to numpy array
    data = np.array(data)
    print("input shape: ", data.shape)
    # normalize input image
    data[0] -= mean[0]
    data[0] /= std[0]
    # create mirrored copy of input
    data2 = np.flip(data, axis=(2))
    # expand dimension of batch size
    data = np.expand_dims(data, axis=0)
    data2 = np.expand_dims(data2, axis=0)
    # predict output
    prediction = model.predict(data)[0,0]
    prediction2 = model.predict(data2)[0,0]
    # mirror the output back
    prediction2 = np.flip(prediction2, axis=(1))
    # load CT image to get SMIR ID, original size, header and afiine
    CT_path = filename  # get right header for SMIR
    CT = nib.load(CT_path)
    # transpose label mat to mask
    prediction = np.mean(np.array([prediction, prediction2]), axis=0)
    label_map_data = np.zeros(prediction.shape, np.int8)
    label_map_data[prediction > config["threshhold"]] = 1
    # write prediction to niftiimage into prediction_path folder
    prediction = Resize(CT.shape, interpolation = "nearest").preprocess(label_map_data)
    predNifti = nib.Nifti1Image(prediction, CT.affine, CT.header)
    print("Output prediction: ", prediction.shape)
    # predNifti.set_data_dtype('short')
    if not os.path.exists(config["prediction_path"]):
        print("create folderrr")
        os.makedirs(config["prediction_path"])
    print("output_filename", output_filename)

    prediction_path = os.path.join(config["prediction_path"], output_filename)
    predNifti.to_filename(prediction_path)

def main():
    dice = []
    # load model and wieghts
    model = ModelLoader(config["model_file"])
    model.load_weights(config["weights_file"])

    input_image = args.i
    output_image = args.o
    print("output_image_shape", output_image)

    # call predict function
    predict(input_image, config["mean"], config["std"], model, output_image)

if __name__ == "__main__":
    main()
