import numpy as np
from sklearn.cluster import KMeans
from pandas import DataFrame

# todo: Calibrate demodulation function for the ONIX


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]
