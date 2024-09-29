"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weights...) these macros will need to be modified accordingly.
"""

from qm.qua import *
from qualang_tools.addons.variables import assign_variables_to_element
import numpy as np
from qualang_tools.analysis import two_state_discriminator
from configuration_mw_fem import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

##############
# QUA macros #
##############

def fit_cosine(x_data, y_data, plot_data=False):
    # Get the number of shots and length of x-axis
    n_shots, x_len = y_data.shape

    # Initialize arrays to store the fitted parameters and analysis results
    amplitudes = np.zeros(n_shots)
    phases = np.zeros(n_shots)
    offsets = np.zeros(n_shots)
    x_plus_pi_half = np.zeros(n_shots)
    x_minus_pi_half = np.zeros(n_shots)

    def cosine_func(x, amplitude, phase):
        return np.abs(amplitude) * np.cos(x + phase)

    # Iterate over each shot
    for i in range(n_shots):
        # Get the y-data for the current shot
        y_data_shot = y_data[i]

        # Center the y-data to oscillate around zero
        y_mean = np.mean(y_data_shot)
        y_data_centered = y_data_shot - y_mean

        # Estimate initial guesses for the fitting parameters
        amplitude_guess = np.max(np.abs(y_data_centered))
        phase_guess = 0

        # Perform the curve fitting with initial guesses
        popt, _ = curve_fit(cosine_func, x_data, y_data_centered, p0=[amplitude_guess, phase_guess])

        # Store the fitted parameters for the current shot
        amplitudes[i] = np.abs(popt[0])
        phases[i] = popt[1]
        offsets[i] = y_mean

        # Find the x-axis value at +π/2 position away from one maxima
        x_plus_pi_half[i] = (np.pi/2 - phases[i])

        # Find the x-axis value at -π/2 position away from one maxima
        x_minus_pi_half[i] = (-np.pi/2 - phases[i])

        # Plot the data and fitted function if plot_data is True
        if plot_data:
            plt.figure()
            plt.plot(x_data, y_data_shot, 'bo', label='Data')
            plt.plot(x_data, cosine_func(x_data, amplitudes[i], phases[i]) + offsets[i], 'r-', label='Fitted Function')

            # Add vertical sticks for x_plus_pi_half and x_minus_pi_half
            plt.axvline(x=x_plus_pi_half[i], color='g', linestyle='--', label='x_plus_pi_half')
            plt.axvline(x=x_minus_pi_half[i], color='m', linestyle='--', label='x_minus_pi_half')
            plt.axvline(x=phases[i], color='k', linestyle='--', label='phase')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Shot {i+1}')
            plt.legend()
            plt.show()

    return amplitudes, phases, offsets, x_plus_pi_half, x_minus_pi_half


def update_LO(config, LO_dict, ifs_vals):
    for qubit_key, LO_value in LO_dict.items():
        # NOTE: different qubits belong to different bands
        if qubit_key in ["q1_xy", "q2_xy", "q3_xy", "q4_xy"]:
            assert (LO_value + ifs_vals[-1] < 5.5e9), "RF = LO+IF value are outside of the band-1 range"
        elif qubit_key in ["q5_xy", "q6_xy", "q7_xy"]:
            assert (LO_value + ifs_vals[0] > 4.5e9) and (LO_value + ifs_vals[-1] < 7.5e9), "RF = LO+IF value are outside of the band-2 range"
        if qubit_key in config["elements"]:
            config["elements"][qubit_key]["MWInput"]["oscillator_frequency"] = LO_value
        else:
            raise KeyError(f"Qubit key '{qubit_key}' not found in the config dictionary.")
    return config


def active_reset(I, I_st, Q, Q_st, state, state_st, resonators, qubits, state_to, delay=None, amplitude=1.0, readout_pulse="midcircuit_readout", weights="rotated_"):

    global_state = declare(int)

    if type(resonators) is not list:
        resonators = [resonators]

    assign(global_state, 0)
    
    for ind, rr in enumerate(resonators):
        # reset_if_phase(rr)
        measure(
            readout_pulse * amp(amplitude),
            rr,
            None,
            dual_demod.full(weights + "cos", weights + "minus_sin", I[ind]),
            dual_demod.full(weights + "sin", weights + "cos", Q[ind]),
        )

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        ###################################################################################
        # NOTE: Ig is promised to be smaller than Ie ONLY if rotation angle is calibrated #
        ###################################################################################

        assign(state[ind], I[ind] > RR_CONSTANTS[rr]["midcircuit_ge_threshold"])
        assign(global_state, (Cast.to_int(state[ind]) << ind) + global_state)

        if state_st is not None:
            save(state[ind], state_st[ind])

    align()

    for ind, qb in enumerate(qubits):

        if delay is None:
            pass
        else:
            wait(delay, qb)
        if state_to == "ground":
            play("x180", qb, condition=state[ind])
        elif state_to == "excited":
            play("x180", qb, condition=~state[ind])
        elif state_to == "none":
            pass

    return global_state


def fit_polynomial(x_values, y_values, order):
    # Fit the polynomial of the given order
    coefficients = np.polyfit(x_values, y_values, order)
    
    # Create a polynomial function from the coefficients
    p = np.poly1d(coefficients)
    
    return coefficients, p


def iq_blobs_analysis(Ig, Qg, Ie, Qe, method):
    """
        It takes I,Q data for ground and excited from save_all() and then
        uses it in three possible ways: Fidelity, SNR, or Overlap
    """
    if method == "fidelity":

        if np.ndim(Ig) == 3:
            tr_Ig = np.transpose(Ig, axes=[1, 2, 0])
            tr_Ie = np.transpose(Ie, axes=[1, 2, 0])
            tr_Qg = np.transpose(Qg, axes=[1, 2, 0])
            tr_Qe = np.transpose(Qe, axes=[1, 2, 0])

            y_dim_len = len(tr_Ig)
            x_dim_len = len(tr_Ig[0])

            angles = []
            thresholds = []
            fidelities = []

            for i in range(y_dim_len):
                angles_col = []
                thresholds_col = []
                fidelities_col = []

                for j in range(x_dim_len):
                    angle_val, threshold_val, fidelity_val, _, _, _, _ = two_state_discriminator(tr_Ig[i, j], tr_Qg[i, j], tr_Ie[i, j], tr_Qe[i, j], False, False)
                    angles_col.append(angle_val)
                    thresholds_col.append(threshold_val)
                    fidelities_col.append(fidelity_val)

                angles.append(angles_col)
                thresholds.append(thresholds_col)
                fidelities.append(fidelities_col)

        elif np.ndim(Ig) == 2:

            tr_Ig = np.transpose(Ig, axes=[1, 0])
            tr_Ie = np.transpose(Ie, axes=[1, 0])
            tr_Qg = np.transpose(Qg, axes=[1, 0])
            tr_Qe = np.transpose(Qe, axes=[1, 0])

            y_dim_len = len(tr_Ig)

            angles = []
            thresholds = []
            fidelities = []

            for i in range(y_dim_len):

                angle_val, threshold_val, fidelity_val, _, _, _, _ = two_state_discriminator(tr_Ig[i], tr_Qg[i], tr_Ie[i], tr_Qe[i], False, False)
                angles.append(angle_val)
                thresholds.append(threshold_val)
                fidelities.append(fidelity_val)

        else:
            angles = []
            thresholds = []
            fidelities = []
            angle_val, threshold_val, fidelity_val, gg_val, ge_val, eg_val, ee_val = two_state_discriminator(Ig, Qg, Ie, Qe, False, False)
            angles.append(angle_val)
            thresholds.append(threshold_val)
            fidelities.append(fidelity_val)
            
        return np.array(angles), np.array(thresholds), np.array(fidelities)
    
    elif method == "snr":

        if np.ndim(Ig) > 0:

            Ig_avg = np.mean(Ig, axis=0)
            Qg_avg = np.mean(Qg, axis=0)
            Ie_avg = np.mean(Ie, axis=0)
            Qe_avg = np.mean(Qe, axis=0)

            Ig_var = np.mean(Ig ** 2, axis=0) - Ig_avg ** 2
            Qg_var = np.mean(Qg ** 2, axis=0) - Qg_avg ** 2
            Ie_var = np.mean(Ie ** 2, axis=0) - Ie_avg ** 2
            Qe_var = np.mean(Qe ** 2, axis=0) - Qe_avg ** 2

            var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4

            Z = (Ie_avg - Ig_avg) + 1j*(Qe_avg - Qg_avg)

            SNR = (np.abs(Z) ** 2) / (2 * var)

        else:
            
            Ig_avg = np.mean(Ig)
            Qg_avg = np.mean(Qg)
            Ie_avg = np.mean(Ie)
            Qe_avg = np.mean(Qe)

            Ig_var = np.mean(Ig ** 2) - Ig_avg ** 2
            Qg_var = np.mean(Qg ** 2) - Qg_avg ** 2
            Ie_var = np.mean(Ie ** 2) - Ie_avg ** 2
            Qe_var = np.mean(Qe ** 2) - Qe_avg ** 2

            var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4

            Z = (Ie_avg - Ig_avg) + 1j*(Qe_avg - Qg_avg)

            SNR = (np.abs(Z) ** 2) / (2 * var)


        return np.array(var), np.array(Z), np.array(SNR)
    
    elif method == "overlap":

        if np.ndim(Ig) == 3:

            tr_Ig = np.transpose(Ig, axes=[1, 2, 0])
            tr_Ie = np.transpose(Ie, axes=[1, 2, 0])
            tr_Qg = np.transpose(Qg, axes=[1, 2, 0])
            tr_Qe = np.transpose(Qe, axes=[1, 2, 0])

            y_dim_len = len(tr_Ig)
            x_dim_len = len(tr_Ig[0])

            overlap = []

            for i in range(y_dim_len):

                overlap_col = []

                for j in range(x_dim_len):

                    min_i = np.min(np.concatenate((tr_Ie[i, j], tr_Ig[i, j])))
                    min_q = np.min(np.concatenate((tr_Qe[i, j], tr_Qg[i, j])))
                    max_i = np.max(np.concatenate((tr_Ie[i, j], tr_Ig[i, j])))
                    max_q = np.max(np.concatenate((tr_Qe[i, j], tr_Qg[i, j])))

                    r = [[min_i, max_i],[min_q, max_q]]

                    P_e, _, _ = np.histogram2d(tr_Ie[i, j], tr_Qe[i, j], bins=35, range=r)
                    P_g, _, _ = np.histogram2d(tr_Ig[i, j], tr_Qg[i, j], bins=35, range=r)

                    overlap_col.append(np.sum(P_e * P_g)/(np.sqrt(np.sum(P_e ** 2)) * np.sqrt(np.sum(P_g ** 2))))
                
                overlap.append(overlap_col)

        elif np.ndim(Ig) == 2:

            tr_Ig = np.transpose(Ig, axes=[1, 0])
            tr_Ie = np.transpose(Ie, axes=[1, 0])
            tr_Qg = np.transpose(Qg, axes=[1, 0])
            tr_Qe = np.transpose(Qe, axes=[1, 0])

            y_dim_len = len(tr_Ig)

            overlap = []

            for i in range(y_dim_len):

                min_i = np.min(np.concatenate((tr_Ie[i], tr_Ig[i])))
                min_q = np.min(np.concatenate((tr_Qe[i], tr_Qg[i])))
                max_i = np.max(np.concatenate((tr_Ie[i], tr_Ig[i])))
                max_q = np.max(np.concatenate((tr_Qe[i], tr_Qg[i])))

                r = [[min_i, max_i],[min_q, max_q]]

                P_e, _, _ = np.histogram2d(tr_Ie[i], tr_Qe[i], bins=35, range=r)
                P_g, _, _ = np.histogram2d(tr_Ig[i], tr_Qg[i], bins=35, range=r)

                overlap.append(np.sum(P_e * P_g)/(np.sqrt(np.sum(P_e ** 2)) * np.sqrt(np.sum(P_g ** 2))))

        else:
            pass

        return np.array(overlap), np.array(overlap), np.array(overlap)
    
def multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators, amplitude=None, weights="", readout_pulse="readout"):
    """Perform multiplexed readout on resonators"""
    if type(resonators) is not list:
        resonators = [resonators]

    for ind, rr in enumerate(resonators):
        if amplitude is None:
            # reset_if_phase(rr)
            measure(
                readout_pulse,
                rr,
                None,
                dual_demod.full(weights + "cos", weights + "sin", I[ind]),
                dual_demod.full(weights + "minus_sin", weights + "cos", Q[ind]),
            )
        else:
            # reset_if_phase(rr)
            measure(
                readout_pulse * amp(amplitude),
                rr,
                None,
                dual_demod.full(weights + "cos", weights + "sin", I[ind]),
                dual_demod.full(weights + "minus_sin", weights + "cos", Q[ind]),
            )

        if state is not None:
            assign(state[ind], I[ind] > RR_CONSTANTS[rr]["ge_threshold"])
        
        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])
        if state_st is not None:
            save(state[ind], state_st[ind])

def qua_declaration(resonators, assign_var_element=False):
    """
    Macro to declare the necessary QUA variables

    :param resonators: Number of qubits used in this experiment
    :return:
    """
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(len(resonators))]
    Q = [declare(fixed) for _ in range(len(resonators))]
    I_st = [declare_stream() for _ in range(len(resonators))]
    Q_st = [declare_stream() for _ in range(len(resonators))]
    if assign_var_element:
        # Workaround to manually assign the results variables to the readout elements
        for ind, rr in enumerate(resonators):
            assign_variables_to_element(rr, I[ind], Q[ind])
    return I, I_st, Q, Q_st, n, n_st


def gef_discriminator_mean_points(Ig, Qg, Ie, Qe, If, Qf, suptitle="qubit 1"):
    """
    Given three blobs in the IQ plane representing g, e, f states,
    finds the averange (mean) point of each blob, classify the label of each data point,
    and computes the confusion matrix of the resulting classification and overall fidelity.
    Plots the raw data of IQ blobs, resulting classification and confusion matrix.

    .. note::
        This function assumes that there are only three blobs in the IQ plane representing gef states (ground, excited, further)
        Unexpected output will be returned in other cases.

    :param float Ig: A vector containing the `I` quadrature of data points in the ground state
    :param float Qg: A vector containing the `Q` quadrature of data points in the ground state
    :param float Ie: A vector containing the `I` quadrature of data points in the excited state
    :param float Qe: A vector containing the `Q` quadrature of data points in the excited state
    :param float If: A vector containing the `I` quadrature of data points in the further excited state
    :param float Qf: A vector containing the `Q` quadrature of data points in the further excited  state
    :param string suptitle: suptitle for the figure
    :returns: A tuple of (fig, Xg_mean, Xe_mean, Xf_mean, fidelity, y_true, y_pred).
        fig - figure handler.
        Xg_mean - average of g state data.
        Xe_mean - average of e state data.
        Xf_mean - average of f state data.
        fidelity - The fidelity for discriminating the states.
        y_true - ground truth of each data point (0: g, 1: e, 2: f).
        y_pred - predicted labels of each data point (0: g, 1: e, 2: f).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    Xg = np.column_stack((Ig, Qg))
    Xe = np.column_stack((Ie, Qe))
    Xf = np.column_stack((If, Qf))
    X = np.concatenate([Xg, Xe, Xf], axis=0)

    # Condition to have the Q equal for both states:
    Xg_mean = Xg.mean(axis=0)
    Xe_mean = Xe.mean(axis=0)
    Xf_mean = Xf.mean(axis=0)
    X_mean = np.column_stack((Xg_mean, Xe_mean, Xf_mean))

    Xg_diff = np.mean((Xg[..., None] - X_mean[None, ...]) ** 2, axis=1)
    Xe_diff = np.mean((Xe[..., None] - X_mean[None, ...]) ** 2, axis=1)
    Xf_diff = np.mean((Xf[..., None] - X_mean[None, ...]) ** 2, axis=1)

    yg_pred = Xg_diff.argmin(axis=1)
    ye_pred = Xe_diff.argmin(axis=1)
    yf_pred = Xf_diff.argmin(axis=1)
    y_pred = np.hstack([yg_pred, ye_pred, yf_pred])
    y_true = np.hstack([np.zeros(Xg.shape[0]), np.ones(Xe.shape[0]), 2 * np.ones(Xf.shape[0])])

    fidelity = (y_true == y_pred).mean() # accuracy of classifier

    def plot_IQ(ax, Xg, Xe, Xf, alpha=1.0, no_legend=False):
        if no_legend:
            ax.scatter(Xg[:, 0], Xg[:, 1], color='r', s=10, alpha=alpha, label="_nolegend_")
            ax.scatter(Xe[:, 0], Xe[:, 1], color='g', s=10, alpha=alpha, label="_nolegend_")
            ax.scatter(Xf[:, 0], Xf[:, 1], color='b', s=10, alpha=alpha, label="_nolegend_")
        else:
            ax.scatter(Xg[:, 0], Xg[:, 1], color='r', s=10, alpha=alpha)
            ax.scatter(Xe[:, 0], Xe[:, 1], color='g', s=10, alpha=alpha)
            ax.scatter(Xf[:, 0], Xf[:, 1], color='b', s=10, alpha=alpha)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        # ax.set_aspect('eq|ual')

    def plot_confusion_matrix(ax, y_true, y_pred, normalize=True):
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plotting
        cax = ax.matshow(cm, cmap='Blues')
        if normalize:
            # Normalize the confusion matrix by row (by the sum of true instances)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Annotate the confusion matrix with text
        for (i, j), val in np.ndenumerate(cm):
            if normalize:
                ax.text(j, i, f'{val:3.2f}', ha='center', va='center', color='black')
            else:
                ax.text(j, i, f'{val}', ha='center', va='center', color='black')

        # Set axis labels and title
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_xticks(np.arange(len(np.unique(y_true))))
        ax.set_yticks(np.arange(len(np.unique(y_true))))
        ax.set_xticklabels(np.unique(y_true))
        ax.set_yticklabels(np.unique(y_true))
        plt.title('Confusion Matrix')


    fig, axss = plt.subplots(2, 3, figsize=(11, 7))
    plt.suptitle(suptitle, fontsize=16)

    # plot all
    ax = axss[0, 0]
    plot_IQ(axss[0, 0], Xg, Xe, Xf, alpha=1.0)
    ax.set_title("g, e, f states")
    ax.legend(["g", "e", "f"])

    # plot
    ax = axss[0, 1]
    plot_IQ(ax, Xg, Xe, Xf, alpha=0.1, no_legend=True)
    ax.scatter(Xg[yg_pred == 0, 0], Xg[yg_pred == 0, 1], color='r', s=12)
    ax.scatter(Xg[yg_pred != 0, 0], Xg[yg_pred != 0, 1], color='k', s=12)
    ax.set_title("g state classification")
    ax.legend(["g", "not g"])

    # plot
    ax = axss[1, 0]
    plot_IQ(ax, Xg, Xe, Xf, alpha=0.1, no_legend=True)
    ax.scatter(Xe[ye_pred == 1, 0], Xe[ye_pred == 1, 1], color='g', s=12)
    ax.scatter(Xe[ye_pred != 1, 0], Xe[ye_pred != 1, 1], color='k', s=12)
    ax.set_title("e state classification")
    ax.legend(["e", "not e"])

    # plot
    ax = axss[1, 1]
    plot_IQ(ax, Xg, Xe, Xf, alpha=0.1, no_legend=True)
    ax.scatter(Xf[yf_pred == 2, 0], Xf[yf_pred == 2, 1], color='b', s=12)
    ax.scatter(Xf[yf_pred != 2, 0], Xf[yf_pred != 2, 1], color='k', s=12)
    ax.set_title("f state classification")
    ax.legend(["f", "not f"])

    ax = axss[0, 2]
    plot_confusion_matrix(ax, y_true, y_pred, normalize=False)
    ax.set_title("confusion matrix [count]")

    ax = axss[1, 2]
    plot_confusion_matrix(ax, y_true, y_pred, normalize=True)
    ax.set_title("confusion matrix [probability]")

    plt.tight_layout()
    return fig, Xg_mean, Xe_mean, Xf_mean, fidelity, y_true, y_pred


def gef_state_discriminator_blob_mean(I, Q, state, state_st, blob_mean):
    s = declare(int)
    dist = declare(fixed, size=3)
    blob_closest = declare(int)
    xs = declare(fixed, size=3)
    ys = declare(fixed, size=3)
    xs2 = declare(fixed, size=3)
    ys2 = declare(fixed, size=3)

    assign(xs[0], I - blob_mean["g"][0])
    assign(xs[1], I - blob_mean["e"][0])
    assign(xs[2], I - blob_mean["f"][0])
    assign(ys[0], Q - blob_mean["g"][1])
    assign(ys[1], Q - blob_mean["e"][1])
    assign(ys[2], Q - blob_mean["f"][1])

    # (0: g, 1: e, 2: f)
    with for_(s, 0, s < 3, s + 1):
        assign(xs2[s], xs[s] * xs[s])
        assign(ys2[s], ys[s] * ys[s])
        assign(dist[s], xs2[s] + ys2[s])

    assign(blob_closest, Math.argmin(dist))
    
    # (0: g, 1: e, 2: f)
    with for_(s, 0, s < 3, s + 1):
        assign(state[s], blob_closest == s)
        save(state[s], state_st)
