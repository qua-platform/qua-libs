from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine
from qm import LoopbackInterface
from qm import SimulationConfig
from bakary import *
from qm.qua import *
import matplotlib.pyplot as plt
from deterministic_configuration import *

b_list = []
for i in range(16):
    with baking(config=config, padding_method="left") as b:
        b.play("gaussOp", "qe1", amp=0.01 * (i+1))
    b_list.append(b)


baked_pulse = [config["waveforms"][f"qe1_baked_wf_I_{i}"]["samples"] for i in range(16)]
# for i in range(len(baked_pulse)):
#     t = np.arange(0, len(baked_pulse[i]), 1)
#     plt.plot(t, baked_pulse[i])
#print(baked_pulse)

QUA_baking_tree = deterministic_run(b_list)
qmm = QuantumMachinesManager("3.122.60.129")
QM = qmm.open_qm(config)
mid = 4


def play_ramsey_tree():
    with if_(j > 7):
        with if_(j > 11):
            with if_(j > 13):
                with if_(j > 14):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")  # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
            with else_():
                with if_(j > 12):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
        with else_():
            with if_(j > 9):
                with if_(j > 10):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
            with else_():
                with if_(j > 8):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
    with else_():
        with if_(j > 3):
            with if_(j > 5):
                with if_(j > 6):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
            with else_():
                with if_(j > 4):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")  # duration Tpihalf+16
        with else_():
            with if_(j > 1):
                with if_(j > 2):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
            with else_():
                with if_(j > 0):
                    wait(4, 'qe1')
                    play("playOp2", "qe1")   # duration Tpihalf+16
                with else_():
                    wait(4, 'qe1')
                    play("playOp2", "qe1")


with program() as prog:
    j = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    with for_(j, 0, j < 16, j+1):

        wait(4, 'qe1')
        # s(j)
        play_ramsey_tree()
        play("playOp", "qe1")
    with for_(j, 0, j < 16, j+1):

        wait(4, 'qe1')
        QUA_baking_tree(j)
        #play_ramsey_tree()
        play("playOp", "qe1")


    #with for_(j, 0, cond=j < 10, update=j + 1):
    # with qua.if_(j > mid):
    #    # b_list[mid + 1].run()
    #     frame_rotation(np.pi, "qe1")
    # with qua.else_():
    #     b_list[mid].run()

job = qmm.simulate(config, prog,
                   SimulationConfig(int(10000//4), simulation_interface=LoopbackInterface(
                       [("con1", 1, "con1", 1)])))  # Use LoopbackInterface to simulate the response of the qubit

samples = job.get_simulated_samples()
plt.figure()
samples.con1.plot()

results = job.result_handles
j = results.j.fetch_all()
j_h_m = results.j_h_m.fetch_all()
j_l_m = results.j_l_m.fetch_all()
#where = results.where.fetch_all()

delay = []

s1 = samples.con1.analog["1"]
s2 = samples.con1.analog["2"]
indices1 = np.nonzero(s1)[0]
indices2 = np.nonzero(s2)[0]
ind1 = []
ind2 = []

for i in range(len(indices1)-1):
    if indices1[i+1] != indices1[i] + 1:
        ind1.append(indices1[i])

for i in range(len(indices2)-1):
    if indices2[i+1] != indices2[i] + 1:
        ind2.append(indices2[i])


for i in range(len(ind2)):
    delay.append(ind2[i]-ind1[i])

print(delay)