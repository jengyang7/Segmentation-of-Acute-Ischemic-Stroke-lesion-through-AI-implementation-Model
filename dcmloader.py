import pydicom as dicom

image_path = "Data/SMIR.dcm"
ds = dicom.dcmread(image_path)

print(ds)