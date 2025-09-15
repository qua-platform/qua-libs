import numpy as np

from sklearn import linear_model


def linear(phi, m, c):
    return m * phi + c

def fit_linear(x_data, y_data):
    """
    Fit data using RANSAC linear regression (ignores y_errors).

    Parameters:
    -----------
    x_data : array-like
        Independent variable data
    y_data : array-like
        Dependent variable data

    Returns:
    --------
    Tuple[float, float, array[bool]]
        line-slope, y-intercept, inlier mask
    """
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    X = np.asarray(x_data)[mask].reshape(-1, 1)
    y = np.asarray(y_data)[mask]

    # Fit with RANSAC
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_

    # Extract coefficients
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    return slope, intercept, inlier_mask


def calculate_crosstalk_coefficient(slope, target_qubit_slope):
    """
    Calculate crosstalk coefficient from linear fit slope.
    
    Parameters:
    -----------
    slope : float
        Linear fit slope (Hz/V)
    slope_error : float
        Error in linear fit slope
    target_qubit_slope : float
        Target qubit slope for chain rule
        
    Returns:
    --------
    Tuple[float, float]
        Crosstalk coefficient and its error
    """
    crosstalk_coefficient = slope / target_qubit_slope
    return crosstalk_coefficient
