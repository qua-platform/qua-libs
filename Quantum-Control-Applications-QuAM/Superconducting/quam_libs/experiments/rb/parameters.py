from dataclasses import dataclass

@dataclass
class BaseNode():
    
    name : str = None 
    parameters: 'Parameters' = None 

@dataclass
class Parameters():
    
    qubit_pair_name: list[str] = None 
    circuit_lengths: tuple[int] = (0,1,2,4)
    num_circuits_per_length: int = 20 
    num_averages: int = 200
    basis_gates: dict[int, str] = None 
    seed: int | None = None
    reset_type: str = "thermal"
    simulate: bool = False
    timeout: int = 100
    
    
    def __post_init__(self):
        if self.basis_gates is None:
            self.basis_gates = {1: 'rz', 2: 'x180', 3 : 'x90', 5: 'cz'}
            
        self.basis_gates_map = dict(zip(self.basis_gates, range(len(self.basis_gates))))