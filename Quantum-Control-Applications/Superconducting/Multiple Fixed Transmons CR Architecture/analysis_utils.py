import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt


def pick_sample(x, y):
    # simple picking, lines, rectangles and text
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title("click on the plot to pick points")
    ax1.plot(x, y)
    ind = plt.ginput(1)
    ax1.axvline(ind[0][0], color="red", lw=2)
    plt.show(block=True)
    return ind[0][0]

def _fit(x, y):

    from scipy import optimize

    def curve_fit3(f, x, y, a0):
        def opt(x, y, a):
            return np.sum(np.abs(f(x, a) - y) ** 2)

        out = optimize.minimize(lambda a: opt(x, y, a), a0)
        return out["x"]

    w = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(x))
    new_w = w[1 : len(freqs // 2)]
    new_f = freqs[1 : len(freqs // 2)]

    ind = new_f > 0
    new_f = new_f[ind]
    new_w = new_w[ind]

    # plt.plot(new_f,np.abs(new_w))
    # plt.show()
    # input("")
    yy = np.abs(new_w)
    first_read_data_ind = np.where(yy[1:] - yy[:-1] > 0)[0][0]  # away from the DC peak

    new_f = new_f[first_read_data_ind:]
    new_w = new_w[first_read_data_ind:]
    # plt.plot(new_f)
    # plt.plot(new_f, np.abs(new_w))
    # plt.show()

    out_freq = new_f[np.argmax(np.abs(new_w))]
    new_w_arg = new_w[np.argmax(np.abs(new_w))]
    ind = np.argmax(np.abs(new_w))
    # plt.plot(out_freq, np.abs(new_w_arg),'o')

    # y_abs = np.abs(new_w)
    x_to_fit_to_gauss = new_f[ind - 2 : ind + 3]
    y_to_fit_to_gauss = new_w[ind - 2 : ind + 3]
    A = np.abs(new_w_arg)
    sigma = (
        np.sum(np.abs(y_to_fit_to_gauss))
        / A
        * (x_to_fit_to_gauss[1] - x_to_fit_to_gauss[0])
    )
    # print(A)
    # print(sigma)
    fit_type = (
        lambda xf, a: A
        * a[0]
        * np.exp(-((xf - out_freq - a[2]) ** 2) / (2 * sigma ** 2 * a[1]))
    )

    popt = curve_fit3(
        fit_type,
        x_to_fit_to_gauss,
        np.abs(y_to_fit_to_gauss),
        [1, 1, 0],
    )
    fit_func = lambda x: fit_type(x, popt)

    # (l,) = plt.plot(x_to_fit_to_gauss, fit_func(x_to_fit_to_gauss),'m')
    # return
    out_freq = out_freq + popt[2]

    omega = out_freq * 2 * np.pi / (x[1] - x[0])  # get gauss for frequency #here
    estimated_angle = np.angle(
        sum(y_to_fit_to_gauss * fit_func(x_to_fit_to_gauss))
        / sum(fit_func(x_to_fit_to_gauss))
    )
    angle0 = estimated_angle - omega * x[0]

    cycle = int(np.ceil(1 / out_freq))
    peaks = (
        np.array(
            [np.std(y[i * cycle : (i + 1) * cycle]) for i in range(int(len(y) / cycle))]
        )
        * np.sqrt(2)
        * 2
    )

    initial_offset = np.mean(y[:cycle])
    cycles_wait = np.where(peaks > peaks[0] * 0.37)[0][-1]

    # amp_guess = max(y[0:cycle])-min(y[0:cycle]) #get gauss for amplitude
    # post_decay = max(y[cycle * cycles_wait : cycle * (cycles_wait+1)])-min(y[cycle * cycles_wait : cycle * (cycles_wait+1)])
    post_decay_mean = np.mean(y[-cycle:])

    # decay_gauss = np.log(amp_guess/post_decay)/(cycles_wait*cycle) / (x[1]-x[0]) #get gauss for decay #here
    decay_gauss = (
        np.log(peaks[0] / peaks[cycles_wait]) / (cycles_wait * cycle) / (x[1] - x[0])
    )  # get gauss for decay #here

    fit_type = lambda x, a: post_decay_mean * a[4] * (
        1 - np.exp(-x * decay_gauss * a[1])
    ) + peaks[0] / 2 * a[2] * (
        np.exp(-x * decay_gauss * a[1])
        * (
            a[5] * initial_offset / peaks[0] * 2
            + np.cos(2 * np.pi * a[0] * omega / (2 * np.pi) * x + a[3])
        )
    )  # here problem, removed the 1+

    # here

    popt = curve_fit3(
        fit_type,
        x,
        y,
        [1, 1, 1, angle0, 1, 1, 1],
    )

    print(
        f"f = {popt[0] * omega / (2 * np.pi)}, phase = {popt[3] % (2 * np.pi)}, tau = {1 / (decay_gauss * popt[1])}, amp = {peaks[0] * popt[2]}, uncertainty population = {post_decay_mean * popt[4]},initial offset = {popt[5] * initial_offset}"
    )
    out = {
        "fit_func": lambda x: fit_type(x, popt),
        "f": popt[0] * omega / (2 * np.pi),
        "phase": popt[3] % (2 * np.pi),
        "tau": 1 / (decay_gauss * popt[1]),
        "amp": peaks[0] * popt[2],
        "uncertainty_population": post_decay_mean * popt[4],
        "initial_offset": popt[5] * initial_offset,
    }
    fit_func = out["fit_func"]
    (l,) = plt.plot(x, fit_func(x), "m", linewidth=2)
    plt.plot(x, fit_type(x, [1, 1, 1, angle0, 1, 1, 1]), "--r", linewidth=1)
    return out


# f = np.load("I_ramsey.npz", allow_pickle=True)
# I = np.array(f.f.arr_0)
# t = np.linspace(16e-9, 10000e-9, len(I))
# plt.plot(t, I)
# out = _fit(t, I)
# plt.grid()
# n = 1  # first peak
# print(out["f"])
# peak_location = (n - out["phase"] / (2 * np.pi)) / out["f"]
# plt.plot(peak_location, out["fit_func"](peak_location), "og")

# f = np.load("rabi_Q_q0.npz", allow_pickle=True)
# I = np.array(f.f.arr_0)
# a = np.linspace(0, 1.0, len(I))
# plt.plot(a, I, '.')
# out = _fit(a, I)
# n = 2  # first peak
# print(out["f"])
# peak_location = (n - out["phase"] / (2 * np.pi)) / out["f"]
# plt.plot(peak_location, out["fit_func"](peak_location), "og")
# print(peak_location)

# f = np.load("rabi_Q_q0.npz", allow_pickle=True)
# I = np.array(f.f.arr_0)
# a = np.linspace(0, 1.0, len(I))
# a_peaked = pick_sample(a, I)
# print(a_peaked)

