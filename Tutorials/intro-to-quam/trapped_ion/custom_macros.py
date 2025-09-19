from qm.qua import *
from quam.core import quam_dataclass
from quam.components.macro import QubitMacro


#############################################################
## Operation macros
#############################################################


@quam_dataclass
class MeasureMacro(QubitMacro):
    threshold: float

    def apply(self):
        # perform shelving operation
        self.qubit.shelving.play("const")
        self.qubit.align()

        # integrating the PMT signal
        I = self.qubit.readout.measure_integrated("const")

        # We declare a QUA variable to store the boolean result of thresholding the I value.
        qubit_state = declare(int)
        # Since |1> is shelved, high fluorescence corresponds to |0>
        # i.e. I < self.threshold implies |1> and vice versa
        assign(qubit_state, Cast.to_int(I < self.threshold))
        return qubit_state


@quam_dataclass
class SingleXMacro(QubitMacro):
    def apply(self, qubit_idx: int):
        self.qubit.ion_displacement.play("ttl")
        align()
        with switch_(qubit_idx):
            with case_(1):
                self.qubit.global_mw.play("x180")
            with case_(2):
                self.qubit.global_mw.play("y180")
                self.qubit.global_mw.play("x180")
                self.qubit.global_mw.play("y180")
        align()
        self.qubit.ion_displacement.play("ttl")
        align()


@quam_dataclass
class DoubleXMacro(QubitMacro):
    def apply(self, qubit_idx: int, amp_scale=1, XX_rep=1):
        i = declare(int)
        self.qubit.ion_displacement.play("ttl")
        align()
        with switch_(qubit_idx):
            with case_(1):
                with for_(i, 0, i < XX_rep * 2, i + 1):
                    self.qubit.global_mw.play("x180", amplitude_scale=amp_scale)
            with case_(2):
                with for_(i, 0, i < XX_rep * 2, i + 1):
                    self.qubit.global_mw.play("y180", amplitude_scale=amp_scale)
                    self.qubit.global_mw.play("x180", amplitude_scale=amp_scale)
                    self.qubit.global_mw.play("y180", amplitude_scale=amp_scale)
        align()
        self.qubit.ion_displacement.play("ttl")
        align()
