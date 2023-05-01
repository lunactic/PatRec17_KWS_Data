""" Compute all features of a given image
"""

import numpy as np
BLACK = 0
WHITE = 1


def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    for counter, window in enumerate(image):
        for counter2, feature in enumerate(window):
            image[counter][counter2] = ((image[counter][counter2]) - mean) / std
    return image

def calculate_features_of_all_samples(training_samples):
    
    return {str(key): calculate_image_features(image) for key, image in training_samples.items()}


def calculate_image_features(image):
    windows = [image[:, i] for i in range(len(image[0]))]
    features = [calculate_window_features(window) for window in windows]
    
    return normalize(np.asarray(features))


def calculate_window_features(window):
    """
    include all features to be calculated in feature_functions
    """
    feature_functions = (upper_contour, lower_contour, b_w_transitions, number_of_black_pixels, fraction_of_black_pixels)
    return [feature_function(window) * 100 for feature_function in feature_functions]


def upper_contour(window):
    """
    :return: the position of the topmost black pixel or 0 if there are none
    """
    for i in range(len(window)):
        if window[i] == BLACK:
            return i/len(window)
    return 1


def lower_contour(window):
    """
    :return: the position of the last black pixel or 0 if there are none
    """
    for i in range(len(window)-1, -1, -1):
        if window[i] == BLACK:
            return i/len(window)
    return 0


def b_w_transitions(window):
    """
    :return: the amount of value transitions in the window independent of their values
    """
    pixel = window[0]
    transitions = 0
    for i in window:
        if i != pixel:
            transitions += 1
            pixel = i
    return transitions/6


def number_of_black_pixels(window):
    number_of_black = 0
    for i in range(len(window)):
        if window[i] == BLACK:
            number_of_black += 1
    return number_of_black/len(window)

def fraction_of_black_pixels(window):
    return number_of_black_pixels(window) / len(window)
