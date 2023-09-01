# %%
# %load_ext autoreload
# %autoreload 2
# %% imports
from qiskit.quantum_info import Pauli
from qiskit.visualization import plot_bloch_vector
from qiskit.result import marginal_counts, sampled_expectation_value
from qiskit.visualization import plot_state_city
from qiskit_experiments.library import StateTomography
from qiskit_experiments.framework import ParallelExperiment
import qiskit
from qiskit.providers.fake_provider import FakePerth
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_paulivec, plot_state_qsphere, plot_distribution
from lib.qua_datasets import integer_histogram
from lib.plot_utils import plot_simulator_output
from qm import SimulationConfig
import matplotlib.pylab as plt
from qm.qua import *
from qw_qm_admin import get_machine, ActiveQubitPair
from qw_qm_admin.quam import QuAM
from lib.tomography import hist_da_to_qiskit_state_tomo_results

#%%

# %% Qua program
shots = 10
with program() as prog:
            n = declare(int)
            basis_q1 = declare(int)
            basis_q2 = declare(int)

            qp_iter = ['q1', 'q2']
            # bases = {q: declare(int), 'mbasis_' + qlabel,
            #                     start=0, stop=3, step=1) for qlabel, q in zip(['q1', 'q2'], qp_iter)}
            bases = {'mbasis_q1': np.array([0, 1, 2]),
                     'mbasis_q2': np.array([0, 1, 2])
            }
            # for basis in bases.values():
            #     basis.declare()

            tomo_amp1 = {q: declare(fixed) for q in qp_iter}
            tomo_amp2 = {q: declare(fixed) for q in qp_iter}
            tomo_amp3 = {q: declare(fixed) for q in qp_iter}
            tomo_amp4 = {q: declare(fixed) for q in qp_iter}

            state1 = declare(bool)
            state2 = declare(bool)
            state = declare(int)

            # machine.all_flux_to_idle()
            # align()
            with for_(n, 0, n < shots, n+1):
                with for_each_(basis_q1,bases['mbasis_q1']):
                    with for_each_(basis_q2,bases['mbasis_q2']):
                        # prep
                        for q, basis in zip(qp_iter, [basis_q0, basis_q1]):
                            with switch_(basis, unsafe=True):
                                with case_(0):  # Z basis
                                    assign(tomo_amp1[q], 0.0)
                                    assign(tomo_amp2[q], 0.0)
                                    assign(tomo_amp3[q], 0.0)
                                    assign(tomo_amp4[q], 0.0)
                                with case_(1):  # X basis
                                    assign(tomo_amp1[q], 0.0)
                                    assign(tomo_amp2[q], -0.5)
                                    assign(tomo_amp3[q], 0.5)
                                    assign(tomo_amp4[q], 0.0)
                                with case_(2):  # Y basis
                                    assign(tomo_amp1[q], 0.5)
                                    assign(tomo_amp2[q], 0.0)
                                    assign(tomo_amp3[q], 0.0)
                                    assign(tomo_amp4[q], 0.5)
                        # qp.q0.initialize_active()      # put back when done simulating!!!
                        # qp.q1.initialize_active()

                        ######################
                        # circuit

                        # align()
                        # qp.q1.xy.play("sy")  # TODO: make sure sign is correct
                        # qp.q0.xy.play("sy")  # TODO: make sure sign is correct
                        # align()
                        # machine.two_qubit_operations.play_cz(
                            # qp.q1.name, qp.q0.name)

                        # align()
                        # qp.q0.xy.play("-sy")
                        # align()

                        # measure
                        # play tomography pulse
                        for q in qp_iter:
                            q.xy.play('x180', amplitude=(
                                tomo_amp1[q], tomo_amp2[q], tomo_amp3[q], tomo_amp4[q]))
                            wait(20, q.xy.name())
                        for q, state_q in [(qp.q0, state0), (qp.q1, state1)]:
                            q.rr.measure_state(state_q)
                        
                    measure(
                    "readout" * amp(a),
                    "rr1",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[0]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[0]),
                )
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])
                        
                        # this corresponds to qiskit little-endian convention
                        # TODO: do we have a macro for this?
                        assign(state, Cast.unsafe_cast_int(state0) +
                               (Cast.unsafe_cast_int(state1) << 1))
                        b.save_all_results()

                        #########################
                        # align()
                        wait(100)

        with stream_processing():
            batches.do_stream_processing()




# %% Calibrate

    simulate = False
    if not simulate:
        job = prog.execute(machine)

    else:
        sim_time = 10000
        job = prog.simulate(SimulationConfig(sim_time))
        plot_simulator_output([[q.xy.name(), q.rr.name(), q.z.name()] for qp in machine.active_qubit_pairs() for q in [qp.q0, qp.q1]],
                              job,
                              prog._config,
                              sim_time * 4)

# %%
ds = prog.get_results(job, batch_label="qp", filen='bell_state_tomo')

dsh = integer_histogram(ds.state, 'shots')
dsh
# %%

for qp in machine.active_qubit_pairs():
    dsh_stacked = dsh.sel(qp=qp_name(qp)).stack(
        mbasis=['mbasis_q0', 'mbasis_q1'])
    qiskit_tomo_res = hist_da_to_qiskit_state_tomo_results(
        dsh_stacked, [qp.q0.name, qp.q1.name])

# try several different visualization methods

plot_state_city(qiskit_tomo_res.analysis_results('state').value,
                title='prepare --, run on KIT', figsize=(10, 10))
# %%
plot_state_hinton(qiskit_tomo_res.analysis_results(
    'state').value, title='prepare --, run on KIT', figsize=(10, 10))
# %%
plot_state_paulivec(qiskit_tomo_res.analysis_results(
    'state').value, title='prepare --, run on KIT', figsize=(10, 10))

# %%
plot_state_qsphere(qiskit_tomo_res.analysis_results(
    'state').value)#, title='prepare --, run on KIT', figsize=(10, 10))

# %%

# %% get expected result with qiskit


backend = AerSimulator.from_backend(FakePerth())


# GHZ State preparation circuit
nq = 2
qc = qiskit.QuantumCircuit(nq)
qc.s(1)
qc.sx(1)
qc.sdg(1)
qc.s(0)
qc.sx(0)
qc.sdg(0)
# qc_ghz.s(0)
qc.cx(1, 0)
# for i in range(1, nq):
# qc.cx(0, i)

# QST Experiment
qstexp1 = StateTomography(qc)
qstdata1 = qstexp1.run(backend, seed_simulation=100).block_for_results()
qstdata1.data()
plot_state_city(qstdata1.analysis_results('state').value,
                title='prepare --, qiskit simulator')
# %% look at the marginals and play around with sampled expectation values


# here the pauli must be diagonal!!
sampled_expectation_value(qiskit_tomo_res.data()[0]['counts'], Pauli('ZZ'))

marginal_counts(qstdata1.data(0)['counts'], [0])

# %%
