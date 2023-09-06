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
# from qw_qm_admin import get_machine, ActiveQubitPair
# from qw_qm_admin.quam import QuAM
from lib.tomography import hist_da_to_qiskit_state_tomo_results
from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results import fetching_tool
import xarray as xr
#%%

# %% Qua program
shots = 10_000
with program() as prog:
            I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
            align()
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
            state_st = declare_stream()
            # the serialization contains this
            # wait((4+(0*(Cast.to_int(v2)+Cast.to_int(v4)))), "rr1")
            # wait((4+(0*(Cast.to_int(v3)+Cast.to_int(v5)))), "rr2") 
            # what does this do??? does it come from the assign to element?

            # machine.all_flux_to_idle()
            # align()
            with for_(n, 0, n < shots, n+1):
                save(n,n_st)
                with for_each_(basis_q1,bases['mbasis_q1']):
                    with for_each_(basis_q2,bases['mbasis_q2']):
                        # prep
                        for q, basis in zip(qp_iter, [basis_q1, basis_q2]):
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

                        # implement active reset on both qubits and remove wait
                        # wait(10*qubit_T1)


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
                            play('x180'*amp(tomo_amp1[q], tomo_amp2[q], tomo_amp3[q], tomo_amp4[q]), f'{q}_xy')
                            wait(20, f'{q}_xy')
                        align()
                        multiplexed_readout(I, None, Q, None, resonators=[1, 2], weights="rotated_")
                        align()
                        assign(state1, (I[0] > ge_threshold_q1))
                        assign(state2, (I[1] > ge_threshold_q2))
                        
                        # this corresponds to qiskit little-endian convention
                        # TODO: do we have a macro for this?
                        assign(state, (Cast.unsafe_cast_int(state1)+(Cast.unsafe_cast_int(state2)<<1)))
                        
                        save(state, state_st)
                        
                        wait(100, )
            with stream_processing():
                state_st.buffer(shots, 3, 3).save("statequbit0_qubit1")
                n_st.save("shot")
# %% Calibrate
qmm = QuantumMachinesManager(host='127.0.0.1', port=8080)#, octave=octave_config)
qm = qmm.open_qm(config)
job = qm.execute(prog)
results = fetching_tool(job, ['shot', 'statequbit0_qubit1' ], mode="wait_for_all")
n , data = results.fetch_all()
#%%
data_vars = {'state': (['qp','shots', 'mbasis_q1', 'mbasis_q2'], [data])}
coords = {'qp':['qubit0_qubit1'],'shots': np.arange(0,shots,1),
          'mbasis_q1': np.array([0, 1, 2]),  'mbasis_q2': np.array([0, 1, 2])}
ds = xr.Dataset(data_vars=data_vars, coords=coords)



# %%
# ds = prog.get_results(job, batch_label="qp", filen='bell_state_tomo')

dsh = integer_histogram(ds.state, 'shots')
dsh
# %%
dsh_stacked = dsh.sel(qp='qubit0_qubit1').stack(mbasis=['mbasis_q1', 'mbasis_q2'])
qiskit_tomo_res = hist_da_to_qiskit_state_tomo_results(dsh_stacked, ['q1', 'q2'])

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
from qm import generate_qua_script
qmm = QuantumMachinesManager(host='127.0.0.1', port=8080)
qm = qmm.open_qm(config)

sourceFile = open('debug.py', 'w')
print(generate_qua_script(prog, config), file=sourceFile)
sourceFile.close()
# %%
