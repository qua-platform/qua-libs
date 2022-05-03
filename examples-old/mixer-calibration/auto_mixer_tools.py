import time
from abc import ABC, abstractmethod
import pyvisa as visa
import scipy.optimize as opti


class AutoMixerCal(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    def get_leakage(self, i0, q0):
        qm.set_dc_offset_by_qe("qubit", "I", i0)
        qm.set_dc_offset_by_qe("qubit", "Q", q0)
        amp_ = self.get_amp()
        return amp_

    def optimize(self):
        start_time = time.time()
        fun_image = lambda x: self.get_leakage(x[0], x[1])
        res_image = opti.minimize(fun_image, [0, 0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 2})
        print(
            f"Image --- g = {res_image.x[0]:.4f}, phi = {res_image.x[1]:.4f} --- "
            f"{int(time.time() - start_time)} seconds --- {int(res_image.fun)} dB"
        )

    @abstractmethod
    def prep_lo(self):
        pass

    @abstractmethod
    def prep_spectrum(self):
        pass

    @abstractmethod
    def prep_image(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def get_lo(self):
        pass

    @abstractmethod
    def get_spectrum(self):
        pass

    @abstractmethod
    def get_amp(self):
        pass

    @abstractmethod
    def __del__(self):
        pass


class FieldFoxAutoCal(AutoMixerCal):
    def __init__(self):
        super().__init__()
        rm = visa.ResourceManager()
        self.conn = rm.open_resource("TCPIP0::192.168.1.9::inst0::INSTR")
        self.timeout = 100000

    # def optimize(self):
    #     start_time = time.time()
    #     fun_image = lambda x: get_leakage(x[0], x[1])
    #     res_image = opti.minimize(fun_image, [0, 0], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 2})
    #     print(f"Image --- g = {res_image.x[0]:.4f}, phi = {res_image.x[1]:.4f} --- "
    #           f"{int(time.time() - start_time)} seconds --- {int(res_image.fun)} dB")

    def get_leakage(i0, q0):
        qm.set_dc_offset_by_qe("qubit", "I", i0)
        qm.set_dc_offset_by_qe("qubit", "Q", q0)
        amp_ = get_amp()
        return amp_

    def get_amp(self):
        self.conn.write("INIT:IMM;*OPC?")
        self.conn.read()
        self.conn.write(f"CALC:MARK1:Y?")
        sig = float(self.conn.read())
        return sig

    def __del__(self):
        self.conn.clear()
        self.conn.close()
