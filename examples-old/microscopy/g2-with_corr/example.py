from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qm import LoopbackInterface
from qm.qua import *
from config import config
import numpy as np
import matplotlib.pyplot as plt

qmm = QuantumMachinesManager(host="127.0.0.1")


def save_vec(vec, name):
    """
    Saves a QUA array to a stream
    :param vec: array to be saved
    :param name: stream name
    :return: void
    """
    i_iter = declare(int)
    with for_(i_iter, 0, i_iter < vec.length(), i_iter + 1):
        save(vec[i_iter], name)


def bin_time_tag_vec(tag_vec, tag_num, bin_size_pow_ns, bin_vec):
    """
    Calculates a histogram of the time tag array from the OPX ADC, bin size (in ns) is 2^(bin_size_pos_ns).
    Note that finding the appropriate bin is done by division in the window size, which is implemented by
    bit shifting the time tag by bin_size_pow_ns.
    :param tag_vec: array of time tags from a QUA measure function
    :param tag_num: number of tags in tag_vec (tag_vec is instantiated to accommodate all possible time tags)
    :param bin_size_pow_ns: log2 of bin size (dividing by a power of 2 simplifies calculation in the OPX)
    :param bin_vec: instantiated array of bins
    :return: void (makes changes to bin_vec)
    """
    i_iter = declare(int)
    bin_ind = declare(int)
    with for_(i_iter, 0, i_iter < tag_num, i_iter + 1):
        assign(bin_ind, tag_vec[i_iter] >> bin_size_pow_ns)
        assign(bin_vec[bin_ind], bin_vec[bin_ind] + 1)


def calc_corr(corr_in1, corr_in2, in_len, fold):
    """
    calculates the correlation between two (equal length) QUA arrays. The output is given by
    out[n] = sum(in1[m+n]*in2[m]), output length is 2*input_len-1
    output is instantiated in the function to allow different size in case of fold
    :fold: if true, calculate one sided  correlation (fold the negative part to the positive part),
     else calculate two sided correlation
    :return: corr_out (makes changes to corr_out)
    """
    i_iter1 = declare(int)
    i_iter2 = declare(int)
    out_ind = declare(int)

    corr_out_len = bin_vec_len if fold else 2 * bin_vec_len - 1
    corr_out = declare(int, size=corr_out_len)

    # calculate for negative m
    with for_(i_iter1, 1, i_iter1 < in_len, i_iter1 + 1):
        with for_(i_iter2, 0, i_iter2 < i_iter1, i_iter2 + 1):
            out_ind = in_len - i_iter1 if fold else i_iter1 - 1
            assign(
                corr_out[out_ind],
                corr_out[out_ind] + (corr_in2[i_iter2] * corr_in1[in_len - i_iter1 + i_iter2]),
            )

    # calculate for m=0 using the QUA dot product function
    out_ind = 0 if fold else in_len - 1
    assign(corr_out[out_ind], Math.dot(corr_in2, corr_in1))

    # calculate positive m
    with for_(i_iter1, 1, i_iter1 < in_len, i_iter1 + 1):
        with for_(i_iter2, 0, i_iter2 < (in_len - i_iter1), i_iter2 + 1):
            out_ind = i_iter1 if fold else i_iter1 + in_len - 1
            assign(
                corr_out[out_ind],
                corr_out[out_ind] + (corr_in2[i_iter2 + i_iter1] * corr_in1[i_iter2]),
            )

    return corr_out


# start of script

# calculate one-sided (fold = True) or two sided correlation (fold=False)
fold = True

# length of measurement as a power of 2
window_size_pow = 9
window_size = int(np.power(2, window_size_pow))

# arrivals need to be shorter by a factor of 8 to enable pulse expansion later
arrivals_size_pow = window_size_pow - 3
arrivals_size = int(np.power(2, arrivals_size_pow))

# generate beam splitter data to validate the script
p = 0.6  # probability of photon emission
# assume arrival event in each time unit are i.i.d. (Geometric inter-arrival times)
arrivals = np.random.binomial(1, p, arrivals_size)
# beam splitter sends an arriving photon to ADC1 or ADC2 with equal probability
splitting_proc = np.random.binomial(1, 0.5, arrivals_size)
photon1 = arrivals * splitting_proc
photon2 = arrivals * (1 - splitting_proc)
# expanding pulses to 8 samples, 4 high, 4 low
photon1 = np.array([[photon1], [np.zeros(len(photon1))]])
photon1 = photon1.flatten(order="F")
photon1 = (photon1.repeat(4) > 0) * 0.25
photon2 = np.array([[photon2], [np.zeros(len(photon2))]])
photon2 = photon2.flatten(order="F")
photon2 = (photon2.repeat(4) > 0) * 0.25

# setting waveforms for loopback
config["waveforms"]["photon_1"]["samples"] = list(photon1)
config["waveforms"]["photon_2"]["samples"] = list(photon2)
config["pulses"]["photon1_pulse"]["length"] = window_size
config["pulses"]["photon2_pulse"]["length"] = window_size

with program() as prog1:
    # measure variables
    # maximal possible size for time tag vec is the window size (assuming arrival at every time unit, JIC)
    time_tag_vec1 = declare(int, size=window_size)
    time_tag_vec2 = declare(int, size=window_size)
    tag_num1 = declare(int)
    tag_num2 = declare(int)

    # calculation variables
    bin_size_pow_ns = 4
    bin_vec_len = int(np.power(2, window_size_pow - bin_size_pow_ns))
    bin_vec1 = declare(int, size=bin_vec_len)
    bin_vec2 = declare(int, size=bin_vec_len)

    # program
    measure(
        "measurement",
        "qe1",
        "stream1",
        time_tagging.raw(time_tag_vec1, window_size, tag_num1),
    )
    measure(
        "measurement",
        "qe2",
        "stream2",
        time_tagging.raw(time_tag_vec2, window_size, tag_num2),
    )

    # binning
    bin_time_tag_vec(time_tag_vec1, tag_num1, bin_size_pow_ns, bin_vec1)
    save_vec(bin_vec1, "bin_vec1")
    bin_time_tag_vec(time_tag_vec2, tag_num2, bin_size_pow_ns, bin_vec2)
    save_vec(bin_vec2, "bin_vec2")

    # correlation
    r12 = calc_corr(bin_vec1, bin_vec2, bin_vec_len, fold)
    save_vec(r12, "corr_output")


# simulate program
job = qmm.simulate(
    config,
    prog1,
    SimulationConfig(
        duration=30000,  # duration of simulation in units of 4ns
        include_analog_waveforms=True,  # include analog waveform names
        include_digital_waveforms=True,  # include digital waveform names
        # loopback to simulate beam splitter response
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)]),
    ),
)

# show negative time axis if correlation is two sided and scale it according to bin size
t_axis = (2**bin_size_pow_ns) * (np.arange(bin_vec_len) if fold else np.arange(-bin_vec_len + 1, bin_vec_len))

corr_output = job.result_handles.get("corr_output").fetch_all()["value"]
plt.plot(t_axis, corr_output)

# validate results
bins_sim1 = job.result_handles.get("bin_vec1").fetch_all()["value"]
bins_sim2 = job.result_handles.get("bin_vec2").fetch_all()["value"]
corr_output_validate = np.correlate(bins_sim2, bins_sim1, mode="full")
if fold:
    # fold negative part to positive part using numpy array slicing
    corr_output_validate_temp = np.zeros(bin_vec_len)
    corr_output_validate_temp[0] = corr_output_validate[bin_vec_len - 1]
    corr_output_validate_temp[1:bin_vec_len] = (
        np.flip(corr_output_validate[0 : bin_vec_len - 1]) + corr_output_validate[bin_vec_len : (2 * bin_vec_len - 1)]
    )
    corr_output_validate = corr_output_validate_temp

plt.plot(t_axis, corr_output_validate, "--")
plt.legend(["simulation", "numpy validation"])
plt.xlabel("$t_{[ns]}$")
plt.ylabel("$R_{12}$")
plt.title("one-sided correlation" if fold else "two-sided correlation")
