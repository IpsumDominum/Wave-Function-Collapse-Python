import numpy as np
import cv2
import random
from collections import defaultdict
import os

output_dir = "outputs"
if os.path.isdir(output_dir):
    pass
else:
    os.makedir(output_dir)
"""
Sample over NxN crops of Input image, to create selection array
for output, each grid selects out of random one of the selection array membe
"""


def cv_img(img, resize=True, wait=True, id=str(random.random())):
    if resize:
        img = cv2.resize(img, (512, 512)) / 255
    cv2.imshow(id, img)
    # if(wait):


def write_img(img, out, resize=True, wait=True, id=str(random.random())):
    if resize:
        img = cv2.resize(img, (512, 512))
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


def wfc_run(
    input_img, N=3, output_size=32, write_output=False, output_name="out_video.avi"
):
    if write_output:
        frame_width = 512
        frame_height = 512
        out = cv2.VideoWriter(
            os.path.join(output_dir, output_name + ".avi"),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            10,
            (frame_width, frame_height),
        )
    else:
        out = None

    coe_matrix = np.ones((output_size, output_size))
    if len(input_img.shape) > 2:
        channels = input_img.shape[2]
    else:
        channels = 1
    input_shape_i = input_img.shape[0] // N
    input_shape_j = input_img.shape[1] // N
    cropped_sets = np.zeros((input_shape_i, input_shape_j, N, N, channels))
    cropped_list = []
    hash_to_idx_dict = {}
    hash_frequency_dict = defaultdict(lambda: 0)
    input_padded = np.pad(input_img, ((N, N), (N, N), (0, 0)), "wrap")
    # Get all subsets of these.
    # Non overlapping tiles

    unique_patterns = []
    valid_neighbours = defaultdict(lambda: [[], [], [], [], [], [], [], []])
    seen_patterns = []
    debug_output = np.zeros((N * 3, N * 3, 3))

    directions_list = [(1, 0), (1, 2), (0, 1), (2, 1), (0, 0), (2, 0), (0, 2), (2, 2)]
    for i in range(input_shape_i):
        for j in range(input_shape_j):
            cropped_sets[i][j][:] = input_img[
                i * N : (i + 1) * N, (j * N) : (j + 1) * N, :
            ]
            hash_code = hash_function(
                input_img[(i * N) : (i + 1) * N, j * N : (j + 1) * N]
            )
            cropped_list.append(hash_code)
            hash_to_idx_dict[hash_code] = (i, j)
            hash_frequency_dict[hash_code] += 1

            debug_output[N : N + N, N : N + N] = input_img[
                i * N : (i + 1) * N, (j * N) : (j + 1) * N, :
            ]

            for index, directions in enumerate(directions_list):
                idx = (i + (directions[0] - 1)) % input_shape_i
                jdx = (j + (directions[1] - 1)) % input_shape_j

                adjacent_img = input_img[
                    idx * N : (idx + 1) * N, (jdx * N) : (jdx + 1) * N, :
                ]

                print(idx * N, (idx + 1) * N, (jdx * N), (jdx + 1) * N)
                print(adjacent_img.shape)
                debug_output[
                    directions[0] * N : directions[0] * N + N,
                    directions[1] * N : directions[1] * N + N,
                ] = adjacent_img
                valid_neighbours[hash_code][index].append(hash_function(adjacent_img))
            cv_img(debug_output)

            k = cv2.waitKey(0)
            if k == ord("q"):
                cv2.destroyAllWindows()
                exit()
    cv2.destroyAllWindows()

    # Do some testing
    # One : Ensure that all patterns are unique
    TEST = False
    if TEST:
        for i in range(len(cropped_list)):
            for j in range(len(cropped_list)):
                if cropped_list[i] == cropped_list[j]:
                    idx = hash_to_idx_dict[cropped_list[i]]
                    jdx = hash_to_idx_dict[cropped_list[j]]
                    assert img_equal(
                        cropped_sets[idx[0]][idx[1]], cropped_sets[jdx[0]][jdx[1]]
                    )
                else:
                    try:
                        assert not img_equal(
                            cropped_sets[idx[0]][idx[1]], cropped_sets[jdx[0]][jdx[1]]
                        )
                    except AssertionError:
                        img_i = cropped_sets[idx]
                        img_j = cropped_sets[jdx]

    # Find all possible patterns
    # Build a table of patterns and their valid neighbours

    # cropped_list = cropped_sets.reshape((input_img.shape[0]*input_img.shape[1],*cropped_sets.shape[2:]))

    # Brute force check all seen patterns

    """
    for nei in valid_neighbours:
        #get some unique neighbours
        for i in range(4):
            valid_neighbours[nei] = np.unique(valid_neighbours[nei])
        #for i in valid_neighbours[nei]:
    """

    def get_pattern(pattern_code):
        idx = hash_to_idx_dict[pattern_code]
        return cropped_sets[idx[0]][idx[1]]

    def get_avg_color(hash_list):
        avg_color = np.zeros((3))
        for key in hash_list:
            idx = hash_to_idx_dict[key]
            avg_color = (
                avg_color + np.mean(cropped_sets[idx[0]][idx[1]].reshape(-1, 3), axis=0)
            ) / 2
        return avg_color

    def render_img(output, output_img, out):
        for i in range(output_size):
            for j in range(output_size):
                if output[i * output_size + j][1] == -1:
                    output_img[i * N : (i + 1) * N, j * N : (j + 1) * N, :] = output[
                        i * output_size + j
                    ][3]
                else:
                    pattern_code = output[i * output_size + j][1]
                    idx = hash_to_idx_dict[pattern_code]
                    # cv_img(cropped_sets[idx[0]][idx[1]])
                    output_img[
                        i * N : (i + 1) * N, j * N : (j + 1) * N, :
                    ] = cropped_sets[idx[0]][idx[1]]
        if write_output:
            write_img(output_img, out)
        cv_img(output_img)

    # For each block we have entropy + pattern

    avg_color = get_avg_color(hash_to_idx_dict.keys())
    output = [
        [
            len(hash_to_idx_dict) + random.random(),
            -1,
            list(hash_to_idx_dict.keys()),
            avg_color,
        ]
        for _ in range(output_size * output_size)
    ]
    output_img = np.ones((output_size * N, output_size * N, 3))

    # Begin by setting every entropy as all possible states
    # output is = [entropy,chosen_img_code,valid_patterns]
    def observe(output):
        # find lowest entropy
        if output[np.argmin(np.array(output)[:, 0])][0] == 10000000:
            return -1
        else:
            return np.argmin(np.array(output)[:, 0])

    def get_probs(code_list):
        prob_list = np.zeros((len(code_list)))
        total_freq = 0
        for idx, item in enumerate(code_list):
            freq = hash_frequency_dict[item]
            prob_list[idx] = 1 / freq
            total_freq += 1 / freq
        return prob_list / total_freq

    def collapse_wave_function(output, index, queue):
        # Go to the index, choose a valid pattern
        # Collapse the possible states of neighbours
        # Choose a pattern based on frequency
        # If there is no valid pattern to choose from, break and say oh no
        # print(output[index][2])

        if len(output[index][2]) == 0:
            output[index][0] = 10000000
            return output
            # output[index][2] = list(hash_to_idx_dict.keys())

        probability_distribution = get_probs(output[index][2])
        output[index][1] = np.random.choice(
            output[index][2], p=probability_distribution
        )
        # output[index][1] = np.random.choice(output[index][2])

        # Set entropy to 0
        output[index][0] = 10000000

        # Collapse the subsequent wave functions
        # up down left right
        # up

        up = (index - output_size) % (output_size * output_size)
        row_dx = up // output_size
        column_dx = up % output_size
        up = row_dx * output_size + column_dx
        # down
        down = (index + output_size) % (output_size * output_size)
        row_dx = down // output_size
        column_dx = down % output_size
        down = row_dx * output_size + column_dx
        # left
        left = (index - 1) % (output_size * output_size)
        row_dx = left // output_size
        column_dx = left % output_size
        left = row_dx * output_size + column_dx
        # right
        right = (index + 1) % (output_size * output_size)
        row_dx = right // output_size
        column_dx = right % output_size
        right = row_dx * output_size + column_dx
        # upleft
        upleft = (index + output_size - 1) % (output_size * output_size)
        row_dx = upleft // output_size
        column_dx = upleft % output_size
        upleft = row_dx * output_size + column_dx
        # upright
        upright = (index + output_size + 1) % (output_size * output_size)
        row_dx = upright // output_size
        column_dx = upright % output_size
        upright = row_dx * output_size + column_dx
        # downleft
        downleft = (index - output_size - 1) % (output_size * output_size)
        row_dx = downleft // output_size
        column_dx = downleft % output_size
        downleft = row_dx * output_size + column_dx
        # downright
        downright = (index - output_size + 1) % (output_size * output_size)
        row_dx = downright // output_size
        column_dx = right % output_size
        downright = row_dx * output_size + column_dx

        # For each, update their valid patterns to only include the ones
        # available in valid neighbours for the chosen index
        for i, direction in enumerate(
            [up, down, left, right, upleft, upright, downleft, downright]
        ):
            # if the direction is already set, we don't have to do anything
            if output[direction][0] == 10000000:
                continue
            # output[index][1] is chosen pattern
            # i is the direction
            # remember up-0 down-1 left-2 right-3 upleft-4 upright-5 downleft-6 downright-7
            # For each direction, the valid moves are those which overlaps with the current allowed
            # adjacencies to the chosen tile.
            new_valid = valid_neighbours[output[index][1]][i]
            # Get the valid neighbours to the direction of the chosen tile.
            check = valid_neighbours[output[index][1]][i]
            for item in list(hash_to_idx_dict.keys()):
                if item in check and item in output[index][2]:
                    new_valid.append(item)
            output[direction][2] = check
            """
            for iaa,valid in enumerate(new_valid):
                idx = hash_to_idx_dict[valid]
                cv_img(cropped_sets[idx],id=str(iaa))
            cv_img(cropped_sets[hash_to_idx_dict[output[index][1]]],id="hi")
            print(i)
            cv2.waitKey(0)
            """
            output[direction][0] = len(new_valid) + random.random()
            # Update avg_color
            output[direction][3] = get_avg_color(output[direction][2])
            # queue.put(direction)

        return output

    from queue import Queue

    queue = Queue()
    # queue.put(observe(output))
    while True:
        index = observe(output)
        if index == -1:
            done = True
        else:
            done = False
        output = collapse_wave_function(output, index, queue)
        render_img(output, output_img, out)
        k = cv2.waitKey(1)
        if k == ord("q") or done == True:
            break
    if write_output:
        out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
