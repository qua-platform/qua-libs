from qm.QuantumMachinesManager import QuantumMachinesManager, QmJob
from qm.qua import *
from qm import SimulationConfig
from configuration import config, qop_ip, save_dir

from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os, fnmatch

from macros import cz_gate
from cosine import Cosine

filename = 'cz_ops_'
modelist = ['sim', 'prev', 'load', 'new']
mode = modelist[int(input("1. simulate, 2. previous job, 3. load data, 4. new run (1-4)?")) - 1]
print("mode: %s" % mode)

dc0_q2 = config["controllers"]["con1"]["analog_outputs"][8]["offset"]
dc0_q1 = config["controllers"]["con1"]["analog_outputs"][7]["offset"]
Phi = np.arange(0, 5, 0.05)  # 5 rotations
n_avg = 1300000

with program() as cz_ops:
    # update_frequency("q2_xy", 0)

    I_g = [declare(fixed) for i in range(2)]
    Q_g = [declare(fixed) for i in range(2)]
    I_e = [declare(fixed) for i in range(2)]
    Q_e = [declare(fixed) for i in range(2)]
    I_st_g = [declare_stream() for i in range(2)]
    Q_st_g = [declare_stream() for i in range(2)]
    I_st_e = [declare_stream() for i in range(2)]
    Q_st_e = [declare_stream() for i in range(2)]
    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    a = declare(fixed)
    phi = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(phi, Phi)):
            # set_dc_offset("q1_z", "single", dc0_q1)

            # 1. Q1 = |0>:
            wait(10000, "q1_z")
            align()
            # Identity
            play("x90_ft", "q2_xy")
            align()
            cz_gate()
            align()
            frame_rotation_2pi(phi, "q2_xy")
            play("x90_ft", "q2_xy")
            align()
            measure("readout" * amp(1.0), "rr1", None,
                    dual_demod.full("rotated_cos", "out1", "rotated_minus_sin", "out2", I_g[0]),
                    dual_demod.full("rotated_sin", "out1", "rotated_cos", "out2", Q_g[0]))
            save(I_g[0], I_st_g[0])
            save(Q_g[0], Q_st_g[0])
            measure("readout" * amp(1.0), "rr2", None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g[1]),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g[1]))
            save(I_g[1], I_st_g[1])
            save(Q_g[1], Q_st_g[1])

            align()

            # 2. Q1 = |1>:
            wait(10000, "q1_z")
            align()
            # Pi-pulse on Q1
            play("x180_ft", "q1_xy")
            align()
            play("x90_ft", "q2_xy")
            align()
            cz_gate()
            align()
            frame_rotation_2pi(phi, "q2_xy")
            play("x90_ft", "q2_xy")
            align()
            measure("readout" * amp(1.0), "rr1", None,
                    dual_demod.full("rotated_cos", "out1", "rotated_minus_sin", "out2", I_e[0]),
                    dual_demod.full("rotated_sin", "out1", "rotated_cos", "out2", Q_e[0]))
            save(I_e[0], I_st_e[0])
            save(Q_e[0], Q_st_e[0])
            measure("readout" * amp(1.0), "rr2", None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e[1]),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e[1]))
            save(I_e[1], I_st_e[1])
            save(Q_e[1], Q_st_e[1])

    with stream_processing():
        # for the progress counter
        n_st.save('n')

        # resonator 1
        I_st_g[0].buffer(len(Phi)).average().save("I1g")
        Q_st_g[0].buffer(len(Phi)).average().save("Q1g")
        I_st_e[0].buffer(len(Phi)).average().save("I1e")
        Q_st_e[0].buffer(len(Phi)).average().save("Q1e")

        # resonator 2
        I_st_g[1].buffer(len(Phi)).average().save("I2g")
        Q_st_g[1].buffer(len(Phi)).average().save("Q2g")
        I_st_e[1].buffer(len(Phi)).average().save("I2e")
        Q_st_e[1].buffer(len(Phi)).average().save("Q2e")

# Prepare Figures:
row, col = 2, 2
fig, ax = plt.subplots(row, col)


def data_present(live=True, data=[]):
    fig.suptitle('Tuning CZ on: %s' % filename, fontsize=20, fontweight='bold', color='blue')

    if len(data) == 0:
        results = fetching_tool(job, ["n", "I1g", "Q1g", "I2g", "Q2g", "I1e", "Q1e", "I2e", "Q2e"], mode="live")
        n, I1g, Q1g, I2g, Q2g, I1e, Q1e, I2e, Q2e = results.fetch_all()
    else:
        n, I1g, Q1g, I2g, Q2g, I1e, Q1e, I2e, Q2e = \
            data.f.n, data.f.I1g, data.f.Q1g, data.f.I2g, data.f.Q2g, data.f.I1e, data.f.Q1e, data.f.I2e, data.f.Q2e
    progress_counter(n, n_avg)

    ax[0, 0].cla()
    ax[0, 0].plot(Phi, I1g, 'b', Phi, I1e, 'r')
    ax[0, 0].set_title('q1 - I , n={}'.format(n))
    ax[1, 0].cla()
    ax[1, 0].plot(Phi, Q1g, 'b', Phi, Q1e, 'r')
    ax[1, 0].set_title('q1 - Q , n={}'.format(n))
    ax[0, 1].cla()
    ax[0, 1].plot(Phi, I2g, 'b', Phi, I2e, 'r')
    ax[0, 1].set_title('q2 - I , n={}'.format(n))
    ax[1, 1].cla()

    try:
        fit = Cosine(Phi, Q2g, plot=False)
        phase_g = fit.out.get('phase')[0]
        ax[1, 1].plot(fit.x_data, fit.fit_type(fit.x, fit.popt) * fit.y_normal, '-b', alpha=0.5)
        fit = Cosine(Phi, Q2e, plot=False)
        phase_e = fit.out.get('phase')[0]
        ax[1, 1].plot(fit.x_data, fit.fit_type(fit.x, fit.popt) * fit.y_normal, '-r', alpha=0.5)

    except Exception as e:
        print(e)

    ax[1, 1].plot(Phi, Q2g, '.b', Phi, Q2e, '.r')
    ax[1, 1].set_title('q2 - Q , n={}, pha_diff={}'.format(n, phase_g - phase_e))

    if live:
        plt.pause(4.0)
    else:
        plt.show()
        if not len(data):
            np.savez(save_dir / filename, n=n, Phi=Phi, I1g=I1g, Q1g=Q1g, I2g=I2g, Q2g=Q2g, I1e=I1e, Q1e=Q1e, I2e=I2e,
                     Q2e=Q2e)
            print("Data saved as %s.npz" % filename)
        plt.close()

    return


# open communication with opx
qmm = QuantumMachinesManager(host=qop_ip, port=80)

if mode == "sim":  # simulate the qua program
    job = qmm.simulate(config, cz_ops, SimulationConfig(15000))
    job.get_simulated_samples().con1.plot()

if mode == "prev":  # check any running previous job
    qm_list = qmm.list_open_quantum_machines()
    qm = qmm.get_qm(qm_list[0])
    print("QM-ID: %s, Queue: %s, Version: %s" % (qm.id, qm.queue.count, qmm.version()))
    job = qm.get_running_job()
    try:
        print("JOB-ID: %s" % job.id())
        while int(input("Continue collecting data (1/0)?")):
            job = QmJob(qmm, job.id())
            data_present(False)
            fig, ax = plt.subplots(row, col)
    except Exception as e:
        print(e)
        qm.close()

if mode == "load":  # load data
    flist = fnmatch.filter(os.listdir(save_dir), 'cz_ops*')
    keyword = input("Enter Keyword if any: ")
    flist = list(filter(lambda x: keyword in x, flist))
    print("Saved data with keyword '%s':\n" % keyword)
    for i, f in enumerate(flist): print("%s. %s" % (i + 1, f))
    f_location = int(input("enter 1-%s: " % len(flist)))
    filename = flist[f_location - 1]
    data = np.load(save_dir / (filename))
    data_present(False, data)

if mode == "new":  # new run
    qm = qmm.open_qm(config)
    job = qm.execute(cz_ops)
    data_present(True)
    interrupt = int(input("Stop execution on closing figure (1/0)?"))
    if interrupt:
        interrupt_on_close(fig, job)
    else:
        print("kill the thread to exit..")
    while job.result_handles.is_processing():
        data_present(True)
    if interrupt: qm.close()

