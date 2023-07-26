from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def power_law(m, a, b, p):
    return a * (p**m) + b


# Generate dummy dataset
x_dummy = np.linspace(start=1, stop=1500, num=50)

# Calculate y-values based on dummy x-values
y_dummy = power_law(x_dummy, 0.46, 0.53, 1 - 1.80e-3)
noise = 0.01 * np.random.normal(size=y_dummy.size)
y_dummy = y_dummy + noise

plt.figure()
plt.plot(x_dummy, y_dummy, ".")
plt.title("Dummy data")
plt.xlabel("Number of Clifford gates")
plt.ylabel("Sequence Fidelity")

pars, cov = curve_fit(
    f=power_law,
    xdata=x_dummy,
    ydata=y_dummy,
    p0=[0.5, 0.5, 1],
    bounds=(-np.inf, np.inf),
    maxfev=2000,
)

plt.plot(x_dummy, power_law(x_dummy, *pars), linestyle="--", linewidth=2)

stdevs = np.sqrt(np.diag(cov))

print("#########################")
print("### Fitted Parameters ###")
print("#########################")
print(f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})")
print("Covariance Matrix")
print(cov)

one_minus_p = 1 - pars[2]
r_c = one_minus_p * (1 - 1 / 2**1)
r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
r_c_std = stdevs[2] * (1 - 1 / 2**1)
r_g_std = r_c_std / 1.875

print("#########################")
print("### Useful Parameters ###")
print("#########################")
print(
    f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1})\n"
    f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1})\n"
    f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
)
