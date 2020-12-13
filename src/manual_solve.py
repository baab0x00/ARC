#!/usr/bin/python

import os
import sys
import json
import numpy as np
import re
from collections import defaultdict

# YOUR CODE HERE: write at least three functions which solve
# specific tasks by transforming the input x and returning the
# result. Name them according to the task ID as in the three
# examples below. Delete the three examples. The tasks you choose
# must be in the data/training directory, not data/evaluation.

###################################################################################
# OpKit: a kit of operations to be (re)used for solving 'hopefully' most problems


def img_create(h, w, color=0):
    return np.full((h, w), fill_value=color, dtype=int)


def img_overlay(base, overlay, pos=(0, 0)):
    underlay = base[pos[0]:pos[0] + overlay.shape[0],
                    pos[1]:pos[1] + overlay.shape[1]]
    overlay = np.where(overlay == -1, underlay, overlay)
    base[pos[0]:pos[0] + overlay.shape[0],
         pos[1]:pos[1] + overlay.shape[1]] = overlay


def img_hist(img):
    d = defaultdict(int)
    for i, j in np.ndindex(img.shape):
        d[img[i, j]] += 1
    return dict(d)


def img_major_color(img):
    hist = img_hist(img)
    return max(hist, key=hist.get)


def img_subimg(img, ulc, lrc):
    return img[ulc[0]:lrc[0]+1, ulc[1]:lrc[1]+1]


def img_interest_area(img, bgc=None):
    mc = img_major_color(img) if bgc is None else bgc
    foreground = np.where(img != mc)
    return (min(foreground[0], default=0), min(foreground[1], default=0)), (max(foreground[0], default=0), max(foreground[1], default=0))


def img_colors(img):
    return set(img_hist(img).keys())


def img_filter(img, color, bg_color=None):
    bgc = bg_color if bg_color is not None else img_major_color(img)
    return np.vectorize(lambda c: c if c == color else bgc)(img)


def img_color_density(img, color):
    fimg = img_filter(img, color)
    ulc, lrc = img_interest_area(fimg)
    area = abs(ulc[0] - lrc[0]) * abs(ulc[1] - lrc[1])
    count = img_hist(img).get(color, 0)
    return count / area


def img_unicolor_objs(img):
    objs_colors = img_colors(img)
    objs_colors.remove(img_major_color(img))
    for obj_color in objs_colors:
        fimg = img_filter(img, obj_color)
        yield img_subimg(fimg, *img_interest_area(fimg))


def img_clear_color(img, color):
    return np.vectorize(lambda c: -1 if c == color else c)(img)

###################################################################################


def solve_6a1e5592(x):
    return x


def solve_b2862040(x):
    return x


def solve_05269061(x):
    return x


def solve_67e8384a(img):
    outimg = img_create(img.shape[0]*2, img.shape[1]*2)

    overlay = img
    img_overlay(outimg, overlay, pos=(0, 0))
    overlay = np.flipud(overlay)
    img_overlay(outimg, overlay, pos=(img.shape[0], 0))
    overlay = np.fliplr(overlay)
    img_overlay(outimg, overlay, pos=(img.shape[0], img.shape[1]))
    overlay = np.flipud(overlay)
    img_overlay(outimg, overlay, pos=(0, img.shape[1]))

    return outimg


def solve_2013d3e2(img):
    obj = img_subimg(img, *img_interest_area(img))
    return obj[:obj.shape[0]//2, :obj.shape[1]//2]


def solve_5ad4f10b(img):

    def _divs(n, m):
        return [1] if m == 1 else (_divs(n, m-1) + ([m] if n % m == 0 else []))

    def divs(n):
        for x in _divs(n, n-1):
            yield x

    bgc = img_major_color(img)
    colors = img_colors(img)
    colors.remove(bgc)
    colors = sorted(
        list(map(lambda c: (c, img_color_density(img, c)), colors)),
        key=lambda e: e[1], reverse=True)
    obj_color = colors[0][0]
    pigment = colors[1][0]

    fimg = img_filter(img, obj_color)
    obj = img_subimg(fimg, *img_interest_area(fimg))

    tile_sides = list(set(list(divs(obj.shape[0])) + list(divs(obj.shape[1]))))
    new_obj = []
    for t in reversed(tile_sides):
        new_obj = img_create(obj.shape[0]//t, obj.shape[1]//t)
        unicolor = True
        for i in range(0, obj.shape[0], t):
            for j in range(0, obj.shape[1], t):
                h = img_hist(img_subimg(obj, (i, j), (i+t-1, j+t-1)))
                unicolor = len(h) == 1
                if unicolor:
                    uc = list(h.keys())[0]
                    new_obj[(i//t), (j//t)] = pigment if uc == obj_color else bgc
                else:
                    break
            if not unicolor:
                break
        if unicolor:
            break

    return new_obj


def solve_c8cbb738(img):
    bgc = img_major_color(img)
    objs = img_unicolor_objs(img)
    objs = list(map(lambda obj: img_clear_color(obj, bgc), objs))
    objs_sizes = (obj.shape for obj in objs)
    combined_size = tuple(map(lambda t: max(t), zip(*objs_sizes)))
    outimg = img_create(*combined_size, bgc)
    for obj in objs:
        centering_pos = tuple(
            (np.array(outimg.shape) - np.array(obj.shape)) // 2)
        img_overlay(outimg, obj, centering_pos)
    return outimg


def solve_681b3aeb(img):

    def conv_overlay(img1, img2):
        for i in range(-img2.shape[0], img1.shape[0] + 1):
            for j in range(-img2.shape[1], img1.shape[1] + 1):
                index = np.array((i, j))
                f = np.where((index < 0), (img2.shape + index),
                             (img1.shape - index))
                intersection = np.array(
                    list(map(min, zip(img1.shape, img2.shape, f))))
                out_size = np.array(img1.shape) + \
                    np.array(img2.shape) - intersection
                outimg = img_create(*out_size, color=-1)
                pos1 = tuple(np.where(index < 0, -index, 0))
                pos2 = tuple(np.where(index < 0, 0, index))
                img_overlay(outimg, img1, pos1)
                img_overlay(outimg, img2, pos2)
                yield outimg

    def does_match(img, objs):
        if(len(img_hist(img).keys()) > 2):
            return False
        else:
            for obj in objs:
                colors = img_hist(obj)
                colors.pop(-1, 0)
                obj_color = list(colors.keys())[0]
                fimg = img_filter(img, color=obj_color, bg_color=-1)
                fobj = img_subimg(fimg, *img_interest_area(fimg, bgc=-1))
                if(not np.array_equal(obj, fobj)):
                    return False

        return True

    bgc = img_major_color(img)
    objs = img_unicolor_objs(img)
    objs = list(map(lambda obj: img_clear_color(obj, bgc), objs))
    for im in conv_overlay(*objs):
        if(does_match(im, objs)):
            return im


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input'])
                   for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output'])
                    for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input'])
                  for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output'])
                   for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))


if __name__ == "__main__":
    main()
