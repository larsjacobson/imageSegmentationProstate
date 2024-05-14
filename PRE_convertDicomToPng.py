
import cv2
import os
import pydicom
import numpy as np

inputdir = 'preTEST/Prostate-MRI-US-Biopsy-1151/01-27-2011-NA-MR PROSTATE WOW CONTRAST-73267/18001.000000-t2spcrstaxial oblProstate-40272/'
outdir = 'test/image/'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

#for f in test_list[:1]:
for f in test_list:   # remove "[:10]" to convert all image
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    #print(img)
    scaled_img = (np.maximum(img, 0) / img.max()) * 255.0
    #print(scaled_img)
    cv2.imwrite(outdir + f.replace('.dcm', '.png'), scaled_img) # write png image


