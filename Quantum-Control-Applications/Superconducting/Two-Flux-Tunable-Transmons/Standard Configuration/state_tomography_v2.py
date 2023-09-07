#%% Qiskit imports
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
#%% QM imports
from qm import SimulationConfig
from qm.qua import *
from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results import fetching_tool
#%% Analysis imports
from tomo_lib.qua_datasets import integer_histogram
from tomo_lib.plot_utils import plot_simulator_output
import matplotlib.pylab as plt
from tomo_lib.tomography import hist_da_to_qiskit_state_tomo_results
import xarray as xr
#%% Parameters
shots = 10_000
qp_iter = ['q1', 'q2'] #names of qubits
qubit_number = len(qp_iter)

# %% QUA program
with program() as prog:
            I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=qubit_number)
            align()
            bases = {}
            basis_qua_var = {}
            for q in qp_iter:
                 basis_qua_var[f'basis_{q}'] = declare(int)
                 bases[f'mbasis_{q}'] =  np.array([0, 1, 2])

            tomo_amp1 = {q: declare(fixed) for q in qp_iter}
            tomo_amp2 = {q: declare(fixed) for q in qp_iter}
            tomo_amp3 = {q: declare(fixed) for q in qp_iter}
            tomo_amp4 = {q: declare(fixed) for q in qp_iter}

            state1 = declare(bool)
            state2 = declare(bool)
            state = declare(int)
            state_st = declare_stream()
            with for_(n, 0, n < shots, n+1):
                save(n,n_st)
                # to generalize the for_each_ loops have to be iterated. not sure how to do it
                with for_each_(basis_qua_var['basis_q1'],bases['mbasis_q1']):
                    with for_each_(basis_qua_var['basis_q2'],bases['mbasis_q2']):

                        for q, basis in zip(qp_iter, [basis_qua_var['basis_q1'], basis_qua_var['basis_q1']]):
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
                        ######################
                        # example circuit
                        # align()
                        # play('x180', 'q1_xy')
                        # play('x180', 'q1_xy')
                        # play tomography pulse
                        for q in qp_iter:
                            play('x180'*amp(tomo_amp1[q], tomo_amp2[q], tomo_amp3[q], tomo_amp4[q]), f'{q}_xy')
                            wait(20, f'{q}_xy')
                        align()
                        # readout
                        multiplexed_readout(I, None, Q, None, resonators=[1, 2], weights="rotated_")
                        align()
                        assign(state1, (I[0] > ge_threshold_q1))
                        assign(state2, (I[1] > ge_threshold_q2))
                        assign(state, (Cast.unsafe_cast_int(state1)+(Cast.unsafe_cast_int(state2)<<1)))
                        save(state, state_st)
                        wait(100, )
                        # implement active reset on both qubits and remove wait
                        # wait(10*qubit_T1)
            with stream_processing():
                state_st.buffer(shots, len(bases['mbasis_q1']), len(bases['mbasis_q1'])).save("statequbit0_qubit1")
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
#%%
dsh = integer_histogram(ds.state, 'shots')
dsh
# %%
dsh_stacked = dsh.sel(qp='qubit0_qubit1').stack(mbasis=['mbasis_q1', 'mbasis_q2'])
qiskit_tomo_res = hist_da_to_qiskit_state_tomo_results(dsh_stacked, ['q1', 'q2'])
#%%
plot_state_city(qiskit_tomo_res.analysis_results('state').value,
                title='prepare --, run on KIT', figsize=(10, 10))











#%%
#######################################################################################################################################

#%%
# try several different visualization methods

# %%
plot_state_hinton(qiskit_tomo_res.analysis_results(
    'state').value, title='experiment', figsize=(10, 10))
# %%
plot_state_paulivec(qiskit_tomo_res.analysis_results(
    'state').value, title='experiment', figsize=(10, 10))

# %%
plot_state_qsphere(qiskit_tomo_res.analysis_results(
    'state').value)#, title='experiment', figsize=(10, 10))

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
                title='simulation')
# %% look at the marginals and play around with sampled expectation values


# here the pauli must be diagonal!!
sampled_expectation_value(qiskit_tomo_res.data()[0]['counts'], Pauli('ZZ'))

marginal_counts(qstdata1.data(0)['counts'], [0])

# %% Serializing for debug
from qm import generate_qua_script
qmm = QuantumMachinesManager(host='127.0.0.1', port=8080)
qm = qmm.open_qm(config)

sourceFile = open('debug.py', 'w')
print(generate_qua_script(prog, config), file=sourceFile)
sourceFile.close()
# %%
