from DCoffsetCalibrator import DCoffsetCalibrator
from TimeDiffCalibrator import TimeDiffCalibrator

from qm.qua import *

import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.constraints import max_norm


class NNStateDiscriminator:
    """
    The state discriminator is a class that generates optimized measure procedure for state discrimination
    of a multi-level qubit.
    .. note:
        Currently only 3-states discrimination is supported. The setup assumed here includes IQ mixer both in the up-
        and down-conversion of the readout pulse.
    """

    def __init__(self, qmm, config: dict, resonators, qubits, calibrate_with, path):
        """
        Constructor for the state discriminator class.
        :param qmm: A QuantumMachineManager object
        :param config:  A quantum machine configuration dictionary with the readout resonators and qubits elements.
                        The RRs and Qubits must be mixed input elements defined in the following way:
                        - all the readout resonators elements should be named "rr0","rr1",...
                        - all the qubits elements should be named "qb0","qb1",...
                        - the index number should go up to rr_num-1
                        - all of the above should be mixedInput elements

        :param resonators: list of the readout resonators elements - list of strings for their names
        :param calibrate_with: list of Quantum elements to use for the calibration of the time delay and the DC offset
        :param path: A folder to save the raw data, training parameters, optimized weights
        """

        self.qmm = qmm
        self.config = config
        self.rr_num = len(resonators)
        self.num_of_states = 3
        self.path = path
        self.resonators = resonators
        self.qubits = qubits
        assert len(self.resonators) == len(self.qubits), 'The number of resonators must equal to the number of qubits'
        self.calibrate_with = calibrate_with
        self._create_dir()
        self.saved_data = None
        self.time_diff = None
        self.MAX_STATES = 300
        self.number_of_raw_data_files = None
        self.final_weights = None
        self.load_config_from_file = True
        self._load_file()

    def _create_dir(self):
        try:
            os.mkdir(self.path)
        except OSError:
            print("Creation of the directory %s failed, it might already exist." % self.path)
        else:
            print("Successfully created the directory %s " % self.path)

    def _load_file(self):
        if os.path.isdir(self.path):
            try:
                file = open(self.path + "\\" + "optimal_params.pkl", 'rb')
                data = pickle.load(file)
                file.close()
                if self.load_config_from_file:
                    self.config = data["config"]
                    print(f"ATTENTION: The configuration was loaded from the file {self.path}\\optimal_params.pkl")
                    print("To use a new imported configuration set: self.load_config_from_file=False\n"
                          "or use a new directory path or delete the optimal_params.pkl file")
                self.final_weights = data["final_weights"]

            except FileNotFoundError:
                pass

    def _training_program(self, prepare_qubits,readout_op, avg_n, states, wait_time):
        with program() as train:
            n = declare(int)
            raw = [declare_stream(adc_trace=True) for i in range(self.rr_num)]
            with for_(n, 0, n < avg_n, n + 1):
                for state in states:
                    # prepare the qubits in the given state
                    prepare_qubits(state, self.qubits)

                    # align all elements
                    align(*self.qubits)

                    # measure the state of all readout resonators SIMULTANEOUSLY using reset phase
                    align(*self.resonators)
                    for i in range(self.rr_num):
                        reset_phase(self.resonators[i])

                        measure("test_readout_pulse_" + str(state[i]), self.resonators[i], raw[i])
                        # TODO - ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # measure(readout_op, self.resonator[i], raw[i])

                    # wait on all elements to relax
                    wait(wait_time, *self.resonators, *self.qubits)

            with stream_processing():
                # save the incoming raw waveform
                for i in range(self.rr_num):
                    raw[i].input1().with_timestamps().save_all('raw1_' + str(i))
                    raw[i].input2().with_timestamps().save_all('raw2_' + str(i))

        return train

    def generate_training_data(self, prepare_qubits, readout_op, avg_n: int, states: list, wait_time: int, calibrate_dc_offset=True,
                               **execute_args):
        """

        :param prepare_qubits:  A program that receives a state of qubits as a list, i.e [0,1,2,1,0,2] and prepares the
                                qubits 0-5 in states [g,e,f,e,g,f], respectively.
                                i.e.  the program will look something like:

                                def prepare_qubit(state,qubits):
                                    align(*qubits)
                                    for i in range(rr_num):
                                        play("prepare"+state[i],qubits[i])

                                where 'qubits' is a list of quantum elements corresponding to the qubits, and
                                "prepare0","prepare1","prepare2" are calibrated pulses corresponding to the preparation
                                of states 0,1,2 (g,e,f) for the different qubits.
        :param readout_op: A string with the name of the readout operation used for reading out the state of the readout
                            resonator. Th operation should have the same name for all resonators.
        :param avg_n: Number of measurement to average, for a larger noise a large averaging number is required

        :param states:  The different combinations of states. A list of lists where each inner list decribes the state
                        of the qubits, i.e,   [[0,2,1,0,2],
                                               [1,0,0,1,2],
                                               [0,1,1,1,2]]
                        The first inner list corresponds to qubit 0 being prepared in state 0(g), qubit 1 being prepared
                        in state 2(f), qubit 2 being prepared in state 1(e), etc.
                        For a large number of qubits the states should be randomized.

        :param wait_time:   The wait time between the preparation and measurement of a certain state of qubits to the
                            next one. The longer it takes for the qubit and for the resonators to relax the longer the
                            wait time should be. wait_time=1 (1 clock cycle) corresponds to a wait time of 4ns.

        :param calibrate_dc_offset: whether to calibrate the DC offset on analog inputs
        :param execute_args: optional QuantumMachine additional execution arguments
        :return:
        """

        if calibrate_dc_offset:
            self._calibrate_dc_offset()

        QM = self.qmm.open_qm(self.config)
        if len(states) > self.MAX_STATES:
            # divide the data and states into chunks of size at most MAX_STATES
            print("ATTENTION: Due to a larger number of states than MAX_STATES the data will be divided into multiple "
                  "files")
            for i in range(len(states) // self.MAX_STATES + 1):

                if i == len(states) // self.MAX_STATES:
                    idx = [i * self.MAX_STATES, len(states)]
                else:
                    idx = [i * self.MAX_STATES, (i + 1) * self.MAX_STATES]

                print(f"Generating data file {i}...")
                job = QM.execute(self._training_program(prepare_qubits, readout_op, avg_n, states[idx[0]:idx[1]], wait_time),
                                 **execute_args)
                job.result_handles.wait_for_all_values()

                print("Writing raw data to file...")
                file = open(self.path + "\\" + f"data_{i}.pkl", 'wb')
                self._save_data(file, job, states[idx[0]:idx[1]], avg_n)
            self.number_of_raw_data_files = len(states) // self.MAX_STATES + 1
        else:
            print("Generating data file 0...")
            job = QM.execute(self._training_program(prepare_qubits, readout_op, avg_n, states, wait_time),
                             **execute_args)
            job.result_handles.wait_for_all_values()

            print("Writing raw data to file...")
            file = open(self.path + "\\" + f"data_0.pkl", 'wb')
            self._save_data(file, job, states, avg_n)
            self.number_of_raw_data_files = 1

    def _calibrate_dc_offset(self):
        already_calibrated = set()
        for element in self.calibrate_with:
            if self.config["elements"][element].get("outputs", False):
                con_name = self.config["elements"][element]["outputs"]["out1"][0]
                if con_name not in already_calibrated:
                    already_calibrated.add(con_name)
                    dc_offsets = DCoffsetCalibrator.calibrate(self.qmm,
                                                              self.config,
                                                              element
                                                              )
                    self.config['controllers'][con_name]['analog_inputs'][1]['offset'] = dc_offsets[con_name][0]
                    self.config['controllers'][con_name]['analog_inputs'][2]['offset'] = dc_offsets[con_name][1]
            else:
                con_name = self.config["elements"][element]["mixInputs"]["I"][0]
                print(f"ATTENTION: Probably tried to calibrate DC offset of '{con_name}' using element '{element}' "
                      f"but no outputs were defined on that element.")

    def _save_data(self, file, job, states, avg_n):
        data = {"raw1_" + str(j): job.result_handles.get("raw1_" + str(j)).fetch_all() for j in range(self.rr_num)}
        data.update(
            {"raw2_" + str(j): job.result_handles.get("raw2_" + str(j)).fetch_all() for j in range(self.rr_num)})
        data.update({"config": self.config})
        data.update({"states": states})
        data.update({"N": avg_n})
        pickle.dump(data, file)
        file.close()

    def _calibrate_time_diff(self, calibrate_dc_offset, **execute_args):
        if calibrate_dc_offset:
            self._calibrate_dc_offset()
        time_diff = {}
        for element in self.calibrate_with:
            if self.config["elements"][element].get("outputs", False):
                time_diff[element] = TimeDiffCalibrator.calibrate(self.qmm,
                                                                  self.config,
                                                                  element,
                                                                  **execute_args
                                                                  )
            else:
                con_name = self.config["elements"][element]["mixInputs"]["I"][0]
                print(f"ATTENTION: Probably tried to calibrate time difference of '{con_name}' using element "
                      f"'{element}' but no outputs were defined on that element.")
        for element in time_diff.keys():
            assert time_diff[self.calibrate_with[0]] == time_diff[element], \
                f"ATTENTION: there's a difference in the time delays between '{self.calibrate_with[0]}' and '{element}' " \
                f"\nMake sure the DC offsets are calibrated."

        self.time_diff = time_diff[self.calibrate_with[0]]

    @staticmethod
    def _quantize_traces(x):
        return np.sum(np.reshape(x, (x.shape[0], -1, 4)), axis=2)

    @staticmethod
    def _down_cos(freq, time_diff, readout_len):
        return np.cos(2 * np.pi * freq * 1e-9 * np.linspace(0 - time_diff, readout_len - 1 - time_diff, readout_len))

    @staticmethod
    def _down_sin(freq, time_diff, readout_len):
        return np.sin(2 * np.pi * freq * 1e-9 * np.linspace(0 - time_diff, readout_len - 1 - time_diff, readout_len))

    def train(self, calibrate_dc_offset=True,
              data_files_idx=None, epochs=1000, kernel_initializer='glorot_uniform', **execute_args
              ):
        """

        :param data_files_idx: Indexes of the data files to train with, i.e [0,1,22,51]

        :param ports:   Contains the controllers and their output IQ port.
                        Assuming that the first port is connected to the first input, and the second output to the
                        second input. And the I and Q ports will be connected to input ports 1 and 2, respectively, for
                        consistency.
                        i.e {"con1":(5,1),"con2":(8,3)} means that output port 5 of con1 is connected to input 1, and
                        output port 1 is connected to input 2, for con2 output port 8 is connected to input 1 and output
                        port 3 is connected to input 2. Then, output port 5 and input port 1 of con1 correspond the I
                        component and output port 1 and input port 2 of con1 correspond to the Q component.

        :param epochs:              Number of training epochs. Could be larger for a better outcome

        :param kernel_initializer:  Initial demodulation weights. One can randomize using the default 'glorot_uniform'
                                    distribution. Another possibility which might work better is to start from a
                                    constant value using tf.keras.initializers.Constant(1)

        :param calibrate_dc_offset: Whether to calibrate the DC offset during the training
        :return:
        """
        if not data_files_idx:
            if not self.number_of_raw_data_files:
                self.number_of_raw_data_files = 1
            data_files_idx = np.arange(self.number_of_raw_data_files)

        self._calibrate_time_diff(calibrate_dc_offset, **execute_args)
        files = [open(self.path + "\\" + f"data_{i}.pkl", 'rb') for i in data_files_idx]
        raws = [pickle.load(a) for a in files]
        for f in files:
            f.close()
        final_weights = []
        models = []

        for j in range(self.rr_num):
            ###################################################
            #                  prepare data                   #
            # - take the average over samples to reduce noise #
            # - combine the data from all the generated files #
            ###################################################
            readout_len = self.config["pulses"]["readout_pulse_" + str(j)]["length"]
            raw1 = np.vstack([np.mean(np.reshape(raw["raw1_" + str(j)]['value']['value'],
                                                 (raw["N"], -1, raw["raw1_" + str(j)]['value']['value'].shape[1])),
                                      axis=0)
                              for raw in raws])
            raw2 = np.vstack([np.mean(np.reshape(raw["raw2_" + str(j)]['value']['value'],
                                                 (raw["N"], -1, raw["raw2_" + str(j)]['value']['value'].shape[1])),
                                      axis=0)
                              for raw in raws])

            labels = np.vstack([[tf.one_hot(s[j], self.num_of_states) for s in raw["states"]] for raw in raws])
            labels = labels[:raw1.shape[0], :]
            freq = self.config["elements"][self.resonators[j]]["intermediate_frequency"]

            # multiply raw input signals by cos/sin with the appropriate frequency
            raw1cos = self._quantize_traces(raw1 * (2 ** -12) * self._down_cos(freq, self.time_diff, readout_len))
            raw1sin = self._quantize_traces(raw1 * (2 ** -12) * self._down_sin(freq, self.time_diff, readout_len))
            raw2cos = self._quantize_traces(raw2 * (2 ** -12) * self._down_cos(freq, self.time_diff, readout_len))
            raw2sin = self._quantize_traces(raw2 * (2 ** -12) * self._down_sin(freq, self.time_diff, readout_len))

            ########################
            # build neural network #
            ########################
            in1sin = tf.keras.Input(shape=(int(readout_len / 4),), name='in1sin')
            in1sin_dense = tf.keras.layers.Dense(1, name="in1sindense", bias_constraint=max_norm(0),
                                                 kernel_initializer=kernel_initializer
                                                 )(in1sin)
            in1cos = tf.keras.Input(shape=(int(readout_len / 4),), name='in1cos')
            in1cos_dense = tf.keras.layers.Dense(1, name="in1cosdense", bias_constraint=max_norm(0),
                                                 kernel_initializer=kernel_initializer
                                                 )(in1cos)
            in1add = tf.keras.layers.Add()([in1cos_dense, in1sin_dense])

            in2sin = tf.keras.Input(shape=(int(readout_len / 4),), name='in2sin')
            in2sin_dense = tf.keras.layers.Dense(1, name="in2sindense", bias_constraint=max_norm(0),
                                                 kernel_initializer=kernel_initializer
                                                 )(in2sin)
            in2cos = tf.keras.Input(shape=(int(readout_len / 4),), name='in2cos')
            in2cos_dense = tf.keras.layers.Dense(1, name="in2cosdense", bias_constraint=max_norm(0),
                                                 kernel_initializer=kernel_initializer
                                                 )(in2cos)
            in2add = tf.keras.layers.Add()([in2cos_dense, in2sin_dense])

            inputs = tf.keras.layers.concatenate([in1add, in2add])

            final = tf.keras.layers.Dense(self.num_of_states, name="final")(inputs)
            model = tf.keras.models.Model(inputs=[in1cos, in1sin, in2cos, in2sin], outputs=final)
            loss_fn = tf.keras.losses.mean_squared_error
            model.compile(optimizer='adam',
                          loss=loss_fn,
                          metrics=['accuracy'])
            tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
            model.fit(
                {"in1sin": raw1sin, "in1cos": raw1cos, "in2sin": raw2sin, "in2cos": raw2cos},
                {"final": labels},
                epochs=epochs,
            )

            model.save(self.path + "\\" + "resonator_model_" + str(j))
            models.append(model)

            ###################
            # extract weights #
            ###################
            cos1_weights = model.get_layer('in1cosdense').get_weights()[0]
            sin1_weights = model.get_layer('in1sindense').get_weights()[0]
            cos2_weights = model.get_layer('in2cosdense').get_weights()[0]
            sin2_weights = model.get_layer('in2sindense').get_weights()[0]
            current_final_weights = model.get_layer('final').get_weights()

            # scale weights to make sure the weights within fixed point 2.19 limits
            scale = np.max(np.abs([cos1_weights.T * raw1cos + sin1_weights.T * raw1sin,
                                   cos1_weights.T * raw1cos + sin1_weights.T * raw1sin,
                                   cos2_weights.T * raw2cos + sin2_weights.T * raw2sin,
                                   cos2_weights.T * raw2cos + sin2_weights.T * raw2sin
                                   ]))
            if scale >= 2:
                cos1_weights = cos1_weights / scale
                sin1_weights = sin1_weights / scale
                cos2_weights = cos2_weights / scale
                sin2_weights = sin2_weights / scale
                current_final_weights = current_final_weights / scale

            ######################################
            # update config with optimal weights #
            ######################################
            # out1
            self.config['integration_weights']['optimal_w1_' + str(j)] = {
                # rounding weights to be in the range of precision of the fixed point
                'cosine': list(np.around(cos1_weights.T[0], 4)),
                'sine': list(np.around(sin1_weights.T[0], 4)),
            }
            # out2
            self.config['integration_weights']['optimal_w2_' + str(j)] = {
                'cosine': list(np.around(cos2_weights.T[0], 4)),
                'sine': list(np.around(sin2_weights.T[0], 4)),
            }

            self.config['pulses']['readout_pulse_' + str(j)]['integration_weights'][
                'optimal_w1_' + str(j)] = 'optimal_w1_' + str(j)
            self.config['pulses']['readout_pulse_' + str(j)]['integration_weights'][
                'optimal_w2_' + str(j)] = 'optimal_w2_' + str(j)

            final_weights.append(current_final_weights)

            # TODO: ATTENTION !!!!!!!!!!!!!!!!!! For testing ONLY - REMOVE when done
            for i in range(self.num_of_states):
                self.config['pulses']['test_readout_pulse_' + str(i)]['integration_weights'][
                    'optimal_w1_' + str(j)] = 'optimal_w1_' + str(j)
                self.config['pulses']['test_readout_pulse_' + str(i)]['integration_weights'][
                    'optimal_w2_' + str(j)] = 'optimal_w2_' + str(j)

        self.final_weights = final_weights
        data = {"config": self.config, "final_weights": final_weights}
        file = open(self.path + "\\" + f"optimal_params.pkl", 'wb')
        pickle.dump(data, file)
        file.close()

    def measure_state(self, state, result, out1, out2, res, temp, adc=None):
        """
        This procedure generates a macro of QUA commands for measuring the readout resonator and discriminating between
        the states of the qubit.
        :param readout_op: A string with the readout operation name for all resonators.
        :param out1: A QUA variable declared as: declare(fixed, size=rr_num)
        :param out2: A QUA variable declared as: declare(fixed, size=rr_num)
        :param res: A QUA variable declared as: declare(fixed, size=num_of_states)
        :param temp: A QUA variable declared as: declare(int)
        :param result: A string for the name of the stream that will receive the discrimination result (0,1 or 2)
        :param adc: (optional) the stream variable which the raw ADC data will be saved and will appear in result
        analysis scope.
        """

        # readout measurement
        align(*self.resonators)
        for i in range(self.rr_num):
            reset_phase(self.resonators[i])
            # measure(readout_op, self.resonators[i], adc,
            # TODO: ATTENTION for testing only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            measure("test_readout_pulse_" + str(state[i]), self.resonators[i], adc,
                    demod.full("optimal_w1_" + str(i), out1[i], "out1"),
                    demod.full("optimal_w2_" + str(i), out2[i], "out2")
                    )
        # state assignment
        for i in range(self.rr_num):
            for j in range(self.num_of_states):
                assign(res[j], self.final_weights[i][0][:, j][0] * out1[i]
                       + self.final_weights[i][0][:, j][1] * out2[i]
                       + self.final_weights[i][1][j] * (2 ** -12))
            assign(temp, Math.argmax(res))
            save(temp, result)
