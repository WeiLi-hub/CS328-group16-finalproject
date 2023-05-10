# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features
def _compute_variance(window):
    return np.var(window, axis=0)
    
def _compute_fft(window):
    freq = np.array(np.fft.rfft(window, axis=0), dtype = float)
    uniq, counts = np.unique(freq, return_counts= True)

    return uniq[counts.argmax()]
    
def get_magnitude(window):
    mag = []
    for signal in window:
        mag.append(np.sqrt(np.sum(signal ** 2)))
    
    return mag

def _compute_entropy(window):
    mag = get_magnitude(window)

    distribution = np.histogram(mag, bins=100)[0]
    prob = distribution / len(mag)

    return -np.sum(prob * np.log(prob, where = prob > 0))

def _compute_peakcounts(window):
    mag = get_magnitude(window)
    peaks = find_peaks(mag, height = 0)

    return len(peaks[0])

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    
    x = []
    feature_names = []
    win = np.array(window)
    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")


    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names
    # variance
    x.append(_compute_variance(win[:,0]))
    feature_names.append("x_variance")

    x.append(_compute_variance(win[:,1]))
    feature_names.append("y_variance")

    x.append(_compute_variance(win[:,2]))
    feature_names.append("z_variance")
    
    # fft
    x.append(_compute_fft(win[:,0]))
    feature_names.append("x_fft")

    x.append(_compute_fft(win[:,1]))
    feature_names.append("y_fft")

    x.append(_compute_fft(win[:,2]))
    feature_names.append("z_fft")

    # entropy of magnitude
    x.append(_compute_entropy(window))
    feature_names.append('entropy')

    # getting number of peaks of magnitude 
    x.append(_compute_peakcounts(window))
    feature_names.append('peakcounts')
    

    feature_vector = list(x)    
    return feature_names, feature_vector