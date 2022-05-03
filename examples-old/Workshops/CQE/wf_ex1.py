import numpy as np
import warnings

from numpy import abs, sqrt

warnings.simplefilter("ignore")
my_func1 = lambda x: 2 * sqrt((-abs(abs(x) - 1)) * abs(3 - abs(x)) / ((abs(x) - 1) * (3 - abs(x)))) * (
    1 + abs(abs(x) - 3) / (abs(x) - 3)
) * sqrt(1 - (x / 7) ** 2) + (5 + 0.97 * (abs(x - 0.5) + abs(x + 0.5)) - 3 * (abs(x - 0.75) + abs(x + 0.75))) * (
    1 + abs(1 - abs(x)) / (1 - abs(x))
)

my_func2 = lambda x: -3 * sqrt(1 - (x / 7) ** 2) * sqrt(abs(abs(x) - 4) / (abs(x) - 4))
my_func3 = lambda x: abs(x / 2) - 0.0913722 * x**2 - 3 + sqrt(1 - (abs(abs(x) - 2) - 1) ** 2)

my_func4 = (
    lambda x: (2.71052 + 1.5 - 0.5 * abs(x) - 1.35526 * sqrt(4 - (abs(x) - 1) ** 2))
    * sqrt(abs(abs(x) - 1) / (abs(x) - 1))
    + 0.9
)


def bottom(x):
    arr = []
    try:
        for val in x:
            v1 = my_func2(val)
            v2 = my_func3(val)
            if not np.isnan(v1):
                arr.append(v1)
            elif not np.isnan(v2):
                arr.append(v2)
    except:
        pass
    return arr


def top(x):
    arr = []
    try:
        for val in x:
            v1 = my_func1(val)
            v2 = my_func4(val)
            if not np.isnan(v1):
                arr.append(v1)
            elif not np.isnan(v2):
                arr.append(v2)
    except:
        pass
    return arr


t = np.linspace(-8, 8, 1000)
top_wf = [0] + (0.1 * np.array(top(t))).tolist() + [0]

bottom_wf = [0] + (0.1 * np.array(bottom(t))).tolist() + [0]
