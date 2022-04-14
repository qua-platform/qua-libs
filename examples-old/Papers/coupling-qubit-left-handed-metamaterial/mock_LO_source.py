class mock_LO_source:
    def __init__(self, freq=0):
        self.name = "LO_source_mock"
        self.freq = freq

    def __repr__(self):
        return f"{self.name} at {self.freq}"

    def set_LO_frequency(self, f):
        self.freq = f

        print(f" frequency set  to {self.freq}")

    def get_LO_frequency(self):
        return self.freq
