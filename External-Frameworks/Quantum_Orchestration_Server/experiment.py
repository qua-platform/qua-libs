# Pick you favorite HTTP communication package
import requests
import matplotlib.pyplot as plt

QMM_host = "theo-4c195fa0.dev.quantum-machines.co"
QMM_port = 443

# standardize communications
def com(command, ip = "127.0.0.1", port = 8000):
    return "http://{}:{}/{}".format(ip, port, command)

def plot_simulation(results):
    plt.figure()
    for con in results.json().keys():
        results.json()[con] = {"analog": {}, "digital": {}}
        for chan in results.json()[con]["analog"].keys():
            plt.plot(results.json()[con]["analog"][chan], label="analog" + chan)
        for chan in results.json()[con]["digital"].keys():
            plt.plot(results.json()[con]["analog"][chan], label="digital" + chan)
    plt.legend()
    plt.xlabel("Time [ns]")
    plt.ylabel("Waveforms [V]")
    plt.title("Simulated waveforms")


# Connect to the QMM
r = requests.get(com('create_qmm'), params={"host": QMM_host, "port": QMM_port})
r.json() # Should print "QMM Created"
# See the current experimental parameters:
p = requests.patch(com('spec/params'))
p.json() # returns a dictionary of parameters
# Partially update the experimental parameters
p = requests.patch(com('spec/params'), params={"df": int(25e3), "n_avg": 10000})
# Check the new values
p.json() # returns a dictionary of parameters
# run the experiment
# requests.put(com('run'), params={"exp": 'spec', "qua_name": "resonator_spec_1D"})
requests.get(com('set_sim_config'), params={"clk_cycles": 1000})
res = requests.get(com('simulate'), params={"exp": 'spec', "qua_name": "resonator_spec_1D"})
plot_simulation(res)
# Check if the job is running
# query_r = requests.get(com("job_running"))
# query_r.json() # will return True if the job is running
# # Get the latest data from the server
# results = requests.get(com('get_data'))
# results.json() # will return a dictionary of the form {I: <I data>, Q: <Q_data>, iteration: <iteration data>}
