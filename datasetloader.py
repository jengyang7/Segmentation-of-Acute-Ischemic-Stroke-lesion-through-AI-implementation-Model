import numpy as np
import cv2
import os
import glob
import nibabel as nib
import numpy as np
import nilearn
from nilearn import plotting


# initialize the list of features and labels
data = []
labels = []
folders = []

for folder in sorted(glob.glob('TRAINING/*/SMIR.Brain.XX.O.CT.*')):
    print(folder)
    for img in glob.glob(os.path.join(folder, "*.nii")):
        print(img)
        img_nii = nib.load(img).get_fdata() # fetch array data instead of proxy image
        print(img_nii.shape)
        data.append(img_nii)
    
# print(data)
# plotting.plot_img("TRAINING/case_05/SMIR.Brain.XX.O.CT.345590/SMIR.Brain.XX.O.CT.345590.nii")
