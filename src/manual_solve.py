#!/usr/bin/python

"""
----------------------------------------------------------------------------------------------------
Student name:   Basm Abd-Rabbo
Student ID:     20235855
GitHub Repo:    https://github.com/baab0x00/ARC
----------------------------------------------------------------------------------------------------

Introduction:
-------------
Even that the main focus of this file (as suggested by the name) is on manual/hand-coded solutions 
for selected ARC problems, I've chosen not to go the whole way towards specialization, The approach 
taken here leans more to generalization and reuseability of small to medium sized set of general-ish 
and basic-ish operations (referred to here as OpKit, short for Operations Kit) which are basically 
an aggressively simplified set of image processing-ish operations (sorry, too many "-ish" in a 
paragraph :) ). Of course the word 'general' in 'general operations' is relative to the topic or the 
domain at hand (which is more or less some sort of image and basic shapes processing/manipulation), 
So consider these set of operations as a simple example of the Domain-Specific Language (DSL) 
referred to by Francois Chollet in 'On the Measure of Intelligence' (https://arxiv.org/abs/1911.01547).

This approach could be (or at least I wish) generalized for an autonomoush AI in which an AI agent 
starts with that set of OpKit (the starting point could accelerate and direct the learning process, 
given that we already know the topic and nature of the problem), and its task would be to compose, 
link, chain, prioritize, and/or augment them during (both) learning and solving the problems.

Please notice the following:
- The image-processing-ish (probably the last -ish) nature of the basic set of operations (OpKit / DSL) 
is reflected somehow on the solutions of the problems, so even if some solutions could have been much 
simpler with much basic operations, they here look higher level like 'filtering', 'histogram', 'overlay', 
etc.

- Because of the same reasons, this approach is definitely not the best in terms of efficiency and 
performance, but I thought that compositionability is more important for this problem than performance.

No special Python libraries nor techniques are used, The main library in action is 'numpy' along side 
basic Python containers/collections including defaultdict, basic Python techniques like generators and 
comprehensions.



Chosen Problems:
----------------
I've chosen at least 3 relatively easy problems, and at least 3 relatively hard ones, just to test 
the hypothesis that the same set of ops would be compositable enough to solve problems on a wide
range of complexity levels.
Relatively easy ones: 67e8384a, 2013d3e2, 6d75e8bb
Relatively hard ones: 5ad4f10b, c8cbb738, 681b3aeb

----------------------------------------------------------------------------------------------------

"""


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
# OpKit: a kit of operations to be (re)used for solving 'hopefully' many problems
###################################################################################
#


def img_create(h, w, color=0):
    ''' Creates a new image of hight `h` and width `w`, filled with `color` '''

    return np.full((h, w), fill_value=color, dtype=int)


def img_overlay(base, overlay, pos=(0, 0)):
    ''' 
    Overlays `overlay` (sub)image onto the `base` image at the position `pos`.
    Note: pos is the coordinates in the `base` image at which the upper left corner of the `overlay`
    would be.
    '''
    underlay = base[pos[0]:pos[0] + overlay.shape[0], pos[1]:pos[1] + overlay.shape[1]]
    overlay = np.where(overlay == -1, underlay, overlay)
    base[pos[0]:pos[0] + overlay.shape[0], pos[1]:pos[1] + overlay.shape[1]] = overlay


def img_hist(img):
    ''' Calculates and returns a histogram of the colors in the image in a form of a dictionary '''

    d = defaultdict(int)
    for i, j in np.ndindex(img.shape):
        d[img[i, j]] += 1
    return dict(d)


def img_major_color(img):
    ''' 
    Calculates the major color in an image, 
    the major color is the one with the highest value in the colors histogram.
    Note: most probably the major color is the background color, this is depended upon a lot in many
    of the solutions.
    '''
    hist = img_hist(img)
    return max(hist, key=hist.get)


def img_subimg(img, ulc, lrc):
    '''
    Creates a subimage of `img` from the upper left corner `ulc` to the lower right corner `lrc`.
    '''
    return img[ulc[0]:lrc[0]+1, ulc[1]:lrc[1]+1]


def img_interesting_area(img, bgc=None):
    '''
    Returns the bounding box (ulc, lrc) of the intersting area in `img`.
    Notes: 
    - The interesting area is defined as the area that just includes all but the background color.
    - If the background color `bgc` is not givin, the function will assume that it is the major color.
    '''
    mc = img_major_color(img) if bgc is None else bgc
    foreground = np.where(img != mc)
    return (min(foreground[0], default=0), min(foreground[1], default=0)), (max(foreground[0], default=0), max(foreground[1], default=0))


def img_colors(img):
    ''' Returns a set of all the colors in the image, including the background and transparent (-1)
    pixels if any '''
    return set(img_hist(img).keys())


def img_filter(img, color, bg_color=None):
    ''' Filters out all colors in `img` except `color`, the filtered out colors are replaced by 
    `bg_color` or the major color if `bg_color` is None. '''
    bgc = bg_color if bg_color is not None else img_major_color(img)
    return np.vectorize(lambda c: c if c == color else bgc)(img)


def img_color_density(img, color):
    ''' Calculates the density of a `color` in an `img`.
    Note: density = the total number of pixels in `color` in the whole image divided by the area of 
    the interesting area containing that color (not the area of the whole image). '''
    fimg = img_filter(img, color)
    ulc, lrc = img_interesting_area(fimg)
    area = abs(ulc[0] - lrc[0]) * abs(ulc[1] - lrc[1])
    count = img_hist(img).get(color, 0)
    return count / area


def img_unicolor_objs(img):
    ''' 
    Returns a generator of all objects that are each composed of a single color.
    Note: this function will treat a single multi colored object as two different ones, adn will 
    separate them into two distinguished outputs.
    '''
    objs_colors = img_colors(img)
    objs_colors.remove(img_major_color(img))
    for obj_color in objs_colors:
        fimg = img_filter(img, obj_color)
        yield img_subimg(fimg, *img_interesting_area(fimg))


def img_clear_color(img, color):
    ''' 
    Returns a copy of `img` with all pixels of color `color` cleared (i.e. made transparent)
    Note: transparent pixels' values = -1 
    '''
    return np.vectorize(lambda c: -1 if c == color else c)(img)

#
###################################################################################




###################################################################################
# Solutions
###################################################################################
#


def solve_67e8384a(img):
    ''' 
    Solves the problem as follows:

    - Consider the input image as a 1/4 tile.
    - Copied and flipped/mirrored down -> right -> up, counter clock wise, in an output image that 
      is 4x (area-wise) the input image.

    Note: All training and test grids are correctly solvable by this solver.
    '''

    # Create output image (4x the input image, area-wise)
    outimg = img_create(img.shape[0]*2, img.shape[1]*2)

    # first copy the tile as it is in the upper left quarter of the output image.
    overlay = img
    img_overlay(outimg, overlay, pos=(0, 0))
    # second vertically flip the tile and overlay it on the lower left quarter.
    overlay = np.flipud(overlay)
    img_overlay(outimg, overlay, pos=(img.shape[0], 0))
    # third, horizontally flip the tile and overlay it on the lower right quarter.
    overlay = np.fliplr(overlay)
    img_overlay(outimg, overlay, pos=(img.shape[0], img.shape[1]))
    #last, vertically flip the tile back and overlay it on the upper right quarter.
    overlay = np.flipud(overlay)
    img_overlay(outimg, overlay, pos=(0, img.shape[1]))

    return outimg


def solve_2013d3e2(img):
    ''' 
    Solves the problem as follows:

    - Extracts the object (which might not be centered in the input image) by distinguishing it from 
      the background as an 'interesting area'.
    - Then cuts the upper left quarter of the extracted object as the repeating tile of the symmetric 
    object, and returns the tile as the solution.

    Note: All training and test grids are correctly solvable by this solver.
    '''

    # first cut the interesting area (area of the object) as a subimage.
    obj = img_subimg(img, *img_interesting_area(img))
    # return the upper left corner (the repeating tile).
    return obj[:obj.shape[0]//2, :obj.shape[1]//2]


def solve_6d75e8bb(img):
    ''' 
    Solves the problem as follows:
    - Extracts the whole rectangle as a single object (subimage of the input image)
    - Replaces all background pixels in the extracted object with the fill color
    - Overlays the painted object on a copy of the input image at exactly the same position as the 
      original object.

    Note: All training and test grids are correctly solvable by this solver.
    '''

    outimg = np.copy(img)
    fill_color = 2
    bgc = img_major_color(img)
    # Identify the bb (Bounding Box) of the object as the interesting area of the image.
    bb = img_interesting_area(img)
    # Exctract the object as a subimage of the input image
    obj = img_subimg(img, *bb)
    # Paint all background pixels in the object subimage with the fill color
    obj = np.where(obj == bgc, fill_color, obj)
    # Overlay the painted object on a copy of the input image at the same position of the original
    # object (the upper left corner of the bounding box)
    img_overlay(outimg, obj, bb[0])

    return outimg


def solve_5ad4f10b(img):
    ''' 
    Solves the problem as follows:
    - Identify all the non-background colors of the image
    - Sort the non-background colors of the image according to the density of each in the image
    - Identify the `object color` as the highest density color, and the `pigment color` as the lower
      density color.
    - Filter the image out of all colors except for the object color (effectly isolating the object
      in its own fram, let's call this image the "object's frame")
    - Cut the object out in a subimage of the object's frame (let's call this one the "object's subimage").
    - Identify the size of the object subimage.
    - Calculate all the divisors of the size of the object subimage.
    - For each `divisor` size:
        - Loop on the the object subimage with a bounding square of side = `divisor`, with a 
          step that is also = `divisor`
            - at each step check if the object subimage has only one color inside the bounding square  
        - If the whole object subimage satisfied the condition that all bounding squares have unique
          colors each, then consider this `divisor` as the "biggest valid divisor"
    - Create a new image (call it the output image) of a size = object subimage size / biggest valid 
      divisor
    - For each bounding square area in the object subimage, paint the corresponding pixel in the output
      image with either the `pigment color` if the corresponding bounding square has an `object color`
      or else a background color

    Note: All training and test grids are correctly solvable by this solver.
    '''

    def _divs(n, m):
        ''' Recursively alculates the the integer divisors of n that are <= m '''
        return [1] if m == 1 else (_divs(n, m-1) + ([m] if n % m == 0 else []))

    def divs(n):
        ''' Calculates all integer divisors of n '''
        for x in _divs(n, n-1):
            yield x

    # Identify the colors of the image
    bgc = img_major_color(img)
    colors = img_colors(img)
    colors.remove(bgc)
    colors = sorted(
        list(map(lambda c: (c, img_color_density(img, c)), colors)),
        key=lambda e: e[1], reverse=True)
    # The object color is the non-background color with the highest density
    obj_color = colors[0][0]
    # The pigment color is the second non-background color after the object's color in density
    pigment = colors[1][0]

    # filter/isolate the object out in its own frame, and subimage it
    fimg = img_filter(img, obj_color)
    obj = img_subimg(fimg, *img_interesting_area(fimg))

    # Calculate all the possible divisors of the size of the object subimage
    tile_sides = list(set(list(divs(obj.shape[0])) + list(divs(obj.shape[1]))))

    new_obj = []
    # for each possible divisor check if the corresponding tile in the object subimage has only one color
    for t in reversed(tile_sides):
        new_obj = img_create(obj.shape[0]//t, obj.shape[1]//t)
        unicolor = True
        for i in range(0, obj.shape[0], t):
            for j in range(0, obj.shape[1], t):
                h = img_hist(img_subimg(obj, (i, j), (i+t-1, j+t-1)))
                # the area has one color if it's color histogram has only one non empty bin
                unicolor = len(h) == 1
                if unicolor:
                    # if the tile has only one color (the uc = unicolor)
                    uc = list(h.keys())[0]
                    # paint the corresponding pixel in the output image
                    new_obj[(i//t), (j//t)] = pigment if uc == obj_color else bgc
                else:
                    break
            if not unicolor:
                break
        if unicolor:
            break

    return new_obj


def solve_c8cbb738(img):
    ''' 
    Solves the problem as follows:
    - Filter all non-background color in its own frame
    - From each frame subimage the object (as the interesting area of the frame)
    - Clear the background color of each object subimage (make it transparent)
    - Calculate the combined size between all objects (widest width, tallest hight)
    - Create a new output image with the calculated combined size
    - Overlay all the objects centered on the created output image

    Note: All training and test grids are correctly solvable by this solver.
    '''

    bgc = img_major_color(img)
    # isolate each object in its own subimage
    objs = img_unicolor_objs(img)
    # clear the background of all the object subimages
    objs = list(map(lambda obj: img_clear_color(obj, bgc), objs))
    # create the output image with the widest width and tallest hight of all object subimages
    objs_sizes = (obj.shape for obj in objs)
    combined_size = tuple(map(lambda t: max(t), zip(*objs_sizes)))
    outimg = img_create(*combined_size, bgc)
    for obj in objs:
        # overlay each object centered on the output image
        centering_pos = tuple(
            (np.array(outimg.shape) - np.array(obj.shape)) // 2)
        img_overlay(outimg, obj, centering_pos)
    return outimg


def solve_681b3aeb(img):

    ''' 
    Solves the problem as follows:
    - Filter all non-background colors each in its own frame
    - From each frame, subimage the object (as the interesting area of the frame)
    - Clear the background color of each object subimage (make it transparent)
    - Convolute one object on the other and test each arrangement until one arrangement 'fits'
        Note: a `Fit Arrangement` is an arrangment of both object in one image in which both 
        the following conditions are true:
          1. There are only two colors in the arrangment, namely the colors of the two objects, (i.e. 
             there is no gaps in the arrangement)
          2. Filtering and subimaging the arrangement for each object color results in an image that 
             is perfectly equivilant to the object's subimage itself (i.e. there is no overlap 
             between the objects in the arranement.)
    - Return the fit arrangment as the output

    Note: All training and test grids are correctly solvable by this solver.
    '''

    def conv_overlay(img1, img2):
        ''' 
        A generator that convolutes img2 over img1 and yields an arrangment on each single 
        convolution step.

        Note: this convolution operation may look unidentical to how convolution may commonly be
        performed in other contexts/senses, the name here refers only to the action of scanning 
        one image over the other, the exact underlying operation(s) are specific to the purpose of
        this specific solution to this specific task.
        '''
        # start with having img2 outside of img1 on the upper left corner
        # scan left to right, and top down, until you end up having img2 outside of img1 on the 
        # lower right corner
        for i in range(-img2.shape[0], img1.shape[0] + 1):
            for j in range(-img2.shape[1], img1.shape[1] + 1):
                # calculate the size of the arrangement
                index = np.array((i, j))
                f = np.where((index < 0), (img2.shape + index), (img1.shape - index))
                intersection = np.array(list(map(min, zip(img1.shape, img2.shape, f))))
                out_size = np.array(img1.shape) + np.array(img2.shape) - intersection

                # create an image for the arrangment
                outimg = img_create(*out_size, color=-1)
                # calculate the position of each object in the arrangement
                pos1 = tuple(np.where(index < 0, -index, 0))
                pos2 = tuple(np.where(index < 0, 0, index))
                # overlay each obj on the arrangement at its position
                img_overlay(outimg, img1, pos1)
                img_overlay(outimg, img2, pos2)
                # yield the arrangment
                yield outimg

    def does_fit(img, objs):
        ''' 
        Checks the fittness of `objs` (the objects) in `img` (the arrangment) 
        An arrangement is fit if and only if both the following conditions are met:
            1. There are only two colors in the arrangment, namely the colors of the two objects, 
               (i.e. there is no gaps in the arrangement)
            2. Filtering and subimaging the arrangement for each object color results in an image 
               that is perfectly equivilant to the object's subimage itself (i.e. there is no 
               overlap between the objects in the arranement.)
        '''
        if(len(img_hist(img).keys()) > 2):
            # The arrangement is not fit if it had more than 2 colors
            return False
        else:
            for obj in objs:
                # create a filtered subimage of the arrangement for each object
                colors = img_hist(obj)
                colors.pop(-1, 0)
                obj_color = list(colors.keys())[0]
                fimg = img_filter(img, color=obj_color, bg_color=-1)
                fobj = img_subimg(fimg, *img_interesting_area(fimg, bgc=-1))
                if(not np.array_equal(obj, fobj)):
                    # if the filtered subimage of the arrangement for the object is different from 
                    # the original object's subimage, this means that some sort of an overlap 
                    # happened between the two objects in the arrangement, i.e. the arrangment is 
                    # not fit.
                    return False
        
        # only a fit arrangment raches this point
        return True

    # identify the background color of the image
    bgc = img_major_color(img)
    # extract and subimage objects in the image
    objs = img_unicolor_objs(img)
    # make the object subimages transparent (clear the background color)
    objs = list(map(lambda obj: img_clear_color(obj, bgc), objs))
    # convolute the two objects
    for arrangement in conv_overlay(*objs):
        # check each arrangement of the convolution
        if(does_fit(arrangement, objs)):
            # return the fit arrangement
            return arrangement

#
###################################################################################




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
