import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


SHOT_AXIS = 0

def snr_map_double_gaussian(map: np.ndarray, shot_axis: int):
    """
    calculates the SNR of the array `map` by fitting a double-gaussian
    distribution to the histogram along the `shot_axis`.
    """
    map = np.moveaxis(map, shot_axis, SHOT_AXIS)

    singlet_mean, singlet_std, triplet_mean, triplet_std = guess_individual_means_stds(map)

    shape = [n for i, n in enumerate(map.shape) if i != SHOT_AXIS]
    fitted_snr = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            n_bins = round(np.sqrt(map.shape[SHOT_AXIS]))
            hist, bins = np.histogram(map[:,i,j], bins=n_bins)
            bins = np.mean(np.vstack([bins[:-1], bins[1:]]), axis=SHOT_AXIS)

            amp_guess = hist.max()

            initial_guess = [amp_guess, singlet_mean[i,j], singlet_std[i,j],
                             amp_guess, triplet_mean[i,j], triplet_std[i,j]]

            # plt.plot(bins, hist)
            # plt.plot(bins, double_gaussian(bins, *initial_guess))

            try:
                popt, pcov = curve_fit(double_gaussian, bins, hist, p0=initial_guess,maxfev=10_000)
                (_, mean_s_fit, std_s_fit, _, mean_t_fit, std_t_fit) = popt
                fitted_snr[i][j] = snr(mean_s_fit, std_s_fit, mean_t_fit, std_t_fit)
                # plt.plot(bins, double_gaussian(bins, *popt))
            except:
                fitted_snr[i][j] = np.nan

            # plt.show()

    return fitted_snr

def snr_map_crude(map: np.ndarray, shot_axis: int):
    """
    calculates the SNR of the array `map` by crudely splitting the distribution
    down the middle, approximating the mean/std of the individual gaussian peaks
    using the mean/std of the remaining, split distributions.
    """
    map = np.moveaxis(map, shot_axis, 0)
    return snr(*guess_individual_means_stds(map))


def snr(singlet_mean: float, singlet_std: float, triplet_mean: float, triplet_std: float):
    """
    calculates the SNR of a double-gaussian by dividing the difference
    in indidividual means by the sum of the individual standard deviatoins
    """
    return (triplet_mean - singlet_mean) / (triplet_std + singlet_std)


def split_singlet_triplet_distributions(map: np.ndarray):
    """
    Returns two distributions from an original array `map` by masking
    values lower/higher than the mean.
    """
    mask = map < map.mean(axis=SHOT_AXIS)

    # crudely split the distributions by thresholding the combined distrubtion
    singlet_dist = map.copy()
    singlet_dist[~mask] = np.nan

    triplet_dist = map.copy()
    triplet_dist[mask] = np.nan

    return singlet_dist, triplet_dist


def guess_individual_means_stds(map: np.ndarray):
    """
    splits `map` along it's mean value to approximate the singlet/triplet
    distributions, returning the mean/std maps of each.
    """
    singlet_dist, triplet_dist = split_singlet_triplet_distributions(map)

    # approximate intra distribution mean/std as mean/std of split distributions
    singlet_mean = np.nanmean(singlet_dist, axis=SHOT_AXIS)
    triplet_mean = np.nanmean(triplet_dist, axis=SHOT_AXIS)

    singlet_std = np.nanstd(singlet_dist, axis=SHOT_AXIS)
    triplet_std = np.nanstd(triplet_dist, axis=SHOT_AXIS)

    return singlet_mean, singlet_std, triplet_mean, triplet_std


def double_gaussian(x, a1, m1, s1, a2, m2, s2):
    return a1 * np.exp(-((x - m1) / s1) ** 2) + a2 * np.exp(-((x - m2) / s2) ** 2)
