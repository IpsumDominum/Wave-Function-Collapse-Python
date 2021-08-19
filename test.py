import cv2
import os
import math
from algos.WFC import wfc_run
from algos.WFC_Overlap import wfc_overlap_run
from algos.random_choice import random_choice_run
from sys import argv

image_dir = os.path.join("samples")
try:
    img_name = argv[1]    
except IndexError:
    print("usage: python test.py --example_image.png")  
    exit()
try:
    write_output = bool(argv[2])
except IndexError:
    write_output = False

item_img = cv2.imread(os.path.join(image_dir,img_name))


try:
    wfc_overlap_run(item_img,2,20,output_name=img_name,GROUND=False,WRITE_VIDEO=True)
except AttributeError as e:
    print("==========================================")
    print("image "+img_name + " not found in ./samples")
    print("Please check that there is a folder named ./samples with images inside")
    print("==========================================")
    print("usage: python test.py --example_image.png")  
    cv2.destroyAllWindows()
    exit()
cv2.destroyAllWindows()

