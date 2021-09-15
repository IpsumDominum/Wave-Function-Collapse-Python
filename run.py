import cv2
import os
import math
from algos.WFC import wfc_run
from algos.WFC_Overlap import wfc_overlap_run
from algos.random_choice import random_choice_run
from sys import argv
import yaml

image_dir = os.path.join("samples")
try:
    spec_file = argv[1]
except IndexError:
    print("specs file not specified, using specs/default.yaml...")
    spec_file = "default.yaml"

if not os.path.isdir("specs"):
    os.makedirs("specs")
else:
    spec_path = os.path.join("specs", spec_file)
with open(spec_path, "r") as stream:
    try:
        specs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(
            "Spec file not found : {} Please ensure that the file is in the specs folder".format(
                spec_path
            )
        )
        exit()


def load_spec(spec_dict, key, default):
    try:
        return specs[spec][key]
    except KeyError:
        print("{} not found, using default value : {}".format(key, default))
        return default


for spec in specs:
    item_img = cv2.imread(os.path.join("samples", spec))
    N = load_spec(specs[spec], "N", 3)
    OUTPUT_SIZE = load_spec(specs[spec], "OUTPUT_SIZE", 3)
    OUTPUT_NAME = load_spec(specs[spec], "OUTPUT_NAME", spec)
    VISUALIZE_ENCODE = load_spec(specs[spec], "VISUALIZE_ENCODE", False)
    VISUALIZE_ADJACENCY = load_spec(specs[spec], "VISUALIZE_ADJACENCY", False)
    WRITE = load_spec(specs[spec], "WRITE", False)
    WRITE_VIDEO = load_spec(specs[spec], "WRITE_VIDEO", False)
    MAX_BACKTRACK = load_spec(specs[spec], "MAX_BACKTRACK", 5)
    GROUND = load_spec(specs[spec], "GROUND", False)
    
    try:
        wfc_overlap_run(
            item_img,
            N,
            output_size=OUTPUT_SIZE,
            output_name=OUTPUT_NAME,
            VISUALIZE_ENCODE=VISUALIZE_ENCODE,
            VISUALIZE_ADJACENCY=VISUALIZE_ADJACENCY,
            MAX_BACKTRACK = MAX_BACKTRACK,
            WRITE=WRITE,
            WRITE_VIDEO=WRITE_VIDEO,
            GROUND = GROUND,
            SPECS=spec
        )
        # wfc_run(item_img,2,20,output_name=img_name,write_output=False)
    except AttributeError as e:
        print("==========================================")
        print("image " + spec + " not found in ./samples")
        print("Please check that there is a folder named ./samples with images inside")
        print("==========================================")
        print("usage: python test.py --example_image.png")
        cv2.destroyAllWindows()
        exit()
    cv2.destroyAllWindows()
    exit()
