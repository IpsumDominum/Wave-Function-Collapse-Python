import cv2
import os
import math
from algos.WFC import wfc_run
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
    
#wfc_run(item_img,5,20,write_output=True,output_name=img_name)
try:
    #wfc_run(item_img,30,20,write_output=False,output_name=img_name)
    wfc_run(item_img,2,20,write_output=write_output,output_name=img_name)
except AttributeError as e:
    print("==========================================")
    print("image "+img_name + " not found in ./samples")
    print("Please check that there is a folder named ./samples with images inside")
    print("==========================================")
    print("usage: python test.py --example_image.png")  
    exit()

cv2.destroyAllWindows()
