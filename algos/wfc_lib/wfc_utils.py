import numpy as np
import cv2
import random
from collections import defaultdict
import os


def cv_wait(time):
    k = cv2.waitKey(time)
    if k == ord("q"):
        cv2.destroyAllWindows()
        exit()
    return k


def cv_img(img, resize=True, wait=True, id=str(random.random())):
    if resize:
        img = cv2.resize(img, (img.shape[0]*10, img.shape[1]*10), interpolation=3) / 255
    cv2.imshow(id, img)
    # if(wait):


def cv_write(img, out, resize=True, wait=True, id=str(random.random())):
    if resize:
        img = cv2.resize(img, (512, 512))
    if out != None:
        out.write(img.astype(np.uint8))


def img_equal(img1, img2):
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            for k in range(img2.shape[2]):
                if img1[i][j][k] != img2[i][j][k]:
                    return False
    return True


def is_same_pattern(pattern1, pattern2):
    # rotate and compare
    if img_equal(pattern1, pattern2):
        return True
    if img_equal(pattern1, np.fliplr(pattern2)):
        return True
    if img_equal(pattern1, np.flipud(pattern2)):
        return True
    for i in range(4):
        img = np.rot90(pattern2)
        if img_equal(pattern1, img):
            return True
        if img_equal(pattern1, np.fliplr(img)):
            return True
        if img_equal(pattern1, np.flipud(img)):
            return True
    return False


def hash_function(img, random_state=0):
    # HASH FUNCTION FROM  Isaac Karth implementation (Modified)
    state = np.random.RandomState(random_state)
    u = img.reshape(np.array(np.prod(img.shape), dtype=np.int64))
    v = state.randint(1 - (1 << 63), 1 << 63, np.prod(img.shape), dtype="int64")
    return np.inner(u, v).astype("int64")


def prepare_write_outputs(output_dir="wfc_outputs", output_name="wfc_output"):
    # prepare_write_outputs(output_dir="wfc_outputs",output_name=output_name) if WRITE_VIDEO else None
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)
    frame_width = 512
    frame_height = 512
    out = cv2.VideoWriter(
        os.path.join(output_dir, output_name + ".avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        80,
        (frame_width, frame_height),
    )
    return out


def get_hash_to_code(pattern_set, ground, hash_frequency_dict, GROUND=False):
    hash_to_code_dict = {}
    pattern_code_set = {}
    code_frequencies = np.zeros((len(pattern_set.keys())))
    avg_color_set = np.zeros((len(pattern_set.keys()), 3))
    for idx, hash_key in enumerate(pattern_set):
        hash_to_code_dict[hash_key] = idx
        pattern_code_set[idx] = pattern_set[hash_key]
        avg_color_set[idx] = np.average(pattern_set[hash_key].reshape(-1, 3), axis=0)
        code_frequencies[idx] = hash_frequency_dict[hash_key]

    ground_coded = {}
    if GROUND:
        for key in ground.keys():
            ground_coded[hash_to_code_dict[key]] = ground[key]
    return (
        pattern_code_set,
        hash_to_code_dict,
        avg_color_set,
        ground_coded,
        code_frequencies,
    )
