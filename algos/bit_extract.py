import os
import cv2
import numpy as np


def not_in(pix, pixel_list):
    for pixel in pixel_list:
        pix_match = True
        for i in range(len(pix)):
            if pix[i] != pixel[i]:
                pix_match = False
                break
        if pix_match == True:
            return False
    return True

# Takes in the name of a sample, outputs the unique pixel bits (colour) in the image.
def unique_colour_extract(filename):
    image_dir = os.path.join("samples")
    image_loc = os.path.join(image_dir, filename)
    image = cv2.imread(image_loc)
    # unique_pixels = np.array([[None,None,None]])         # efficiency > memory
    unique_pixels = np.array([image[0,0]])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pix = image[i, j]
            if not_in(pix, unique_pixels):
                unique_pixels = np.append(unique_pixels, [pix], axis=0)
    return unique_pixels

print(unique_colour_extract("mooncity4.png"))




