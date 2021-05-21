import shutil
import numpy as np
from .sitk_utils import resample_to_spacing, calculate_origin_offset

class Resize:
    def __init__(self, new_image_shape, interpolation="linear"): # interpolation can be ‘continuous’, ‘linear’, or ‘nearest’. we use linear as default
        self.new_image_shape = new_shape
        self.interpolation = interpolation

    def preprocess(self, image,):
        one_shape = (1.,1.,1.)
        new_image_spacing = np.divide(one_shape, np.divide(self.new_image_shape, image.shape))
        new_image = resample_to_spacing(image, one_shape, new_image_spacing, interpolation=self.interpolation)
        return new_image