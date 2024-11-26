import numpy as np

c1_table_XZ = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23],
       [ 1,  0,  3,  2, 16,  7, 17,  5, 23, 22, 14, 15, 13, 12, 10, 11,
         4,  6, 21, 20, 19, 18,  9,  8],
       [ 2,  3,  0,  1, 17, 13, 16, 12, 22, 23, 11, 10,  7,  5, 15, 14,
         6,  4, 20, 21, 18, 19,  8,  9],
       [ 3,  2,  1,  0,  6, 12,  4, 13,  9,  8, 15, 14,  5,  7, 11, 10,
        17, 16, 19, 18, 21, 20, 23, 22],
       [ 4, 17, 16,  6,  0, 21,  3, 19, 14, 11, 22,  9, 20, 18,  8, 23,
         2,  1, 13,  7, 12,  5, 10, 15],
       [ 5,  7, 12, 13, 22,  0,  8,  1,  6, 16, 21, 20,  2,  3, 18, 19,
         9, 23, 14, 15, 11, 10,  4, 17],
       [ 6, 16, 17,  4,  3, 20,  0, 18, 11, 14, 23,  8, 21, 19,  9, 22,
         1,  2,  7, 13,  5, 12, 15, 10],
       [ 7,  5, 13, 12,  9,  1, 23,  0, 17,  4, 18, 19,  3,  2, 21, 20,
        22,  8, 10, 11, 15, 14, 16,  6],
       [ 8,  9, 23, 22, 13, 11,  5, 14, 20, 18, 17,  6, 10, 15, 16,  4,
         7, 12,  1,  3,  0,  2, 19, 21],
       [ 9,  8, 22, 23,  7, 14, 12, 11, 21, 19, 16,  4, 15, 10, 17,  6,
        13,  5,  2,  0,  3,  1, 18, 20],
       [10, 15, 11, 14, 21, 22, 18, 23, 13, 12,  0,  2,  9,  8,  3,  1,
        20, 19,  6, 17, 16,  4,  5,  7],
       [11, 14, 10, 15, 19,  8, 20,  9,  5,  7,  2,  0, 23, 22,  1,  3,
        18, 21, 16,  4,  6, 17, 13, 12],
       [12, 13,  5,  7, 23,  3,  9,  2,  4, 17, 20, 21,  1,  0, 19, 18,
         8, 22, 11, 10, 14, 15,  6, 16],
       [13, 12,  7,  5,  8,  2, 22,  3, 16,  6, 19, 18,  0,  1, 20, 21,
        23,  9, 15, 14, 10, 11, 17,  4],
       [14, 11, 15, 10, 18,  9, 21,  8, 12, 13,  1,  3, 22, 23,  2,  0,
        19, 20, 17,  6,  4, 16,  7,  5],
       [15, 10, 14, 11, 20, 23, 19, 22,  7,  5,  3,  1,  8,  9,  0,  2,
        21, 18,  4, 16, 17,  6, 12, 13],
       [16,  6,  4, 17,  1, 18,  2, 20, 10, 15,  9, 22, 19, 21, 23,  8,
         3,  0, 12,  5, 13,  7, 14, 11],
       [17,  4,  6, 16,  2, 19,  1, 21, 15, 10,  8, 23, 18, 20, 22,  9,
         0,  3,  5, 12,  7, 13, 11, 14],
       [18, 20, 19, 21, 14, 16, 10,  6,  2,  3,  7, 13,  4, 17, 12,  5,
        15, 11, 23,  8, 22,  9,  1,  0],
       [19, 21, 18, 20, 11, 17, 15,  4,  1,  0, 13,  7,  6, 16,  5, 12,
        10, 14, 22,  9, 23,  8,  2,  3],
       [20, 18, 21, 19, 15,  6, 11, 16,  0,  1, 12,  5, 17,  4,  7, 13,
        14, 10,  9, 22,  8, 23,  3,  2],
       [21, 19, 20, 18, 10,  4, 14, 17,  3,  2,  5, 12, 16,  6, 13,  7,
        11, 15,  8, 23,  9, 22,  0,  1],
       [22, 23,  9,  8,  5, 10, 13, 15, 18, 20,  4, 16, 11, 14,  6, 17,
        12,  7,  3,  1,  2,  0, 21, 19],
       [23, 22,  8,  9, 12, 15,  7, 10, 19, 21,  6, 17, 14, 11,  4, 16,
         5, 13,  0,  2,  1,  3, 20, 18]])

# %%
import numpy as np

gates_list = [
    "I",	
    "X180",
    "Z-90 X180 Z90",
    "X90 Z180 Z90",
    
    "X90 Z_90 X90 Z90",
    "X90 Z90 X90 Z_90",
    "X_90 Z_90 X90 Z90",
    "X_90 Z90 X90 Z_90",
    "Z_90 X90 Z90 X90",
    "Z_90 X90 Z90 X_90",
    "Z90 X90 Z_90 X90",
    "Z90 X90 Z_90 X_90",
    
    "X90",
    "X-90",
    "Z_90 X90 Z90",
    "Z90 X90 Z_90",
    "Z90",
    "Z-90",
    
    "X90 Z_90 X90",
    "X90 Z90 X90",
    "Z_90 X90 Z90 X90 Z90",
    "Z_90 X90 Z90 X_90 Z_90",
    "X180 Z90",
    "X180 Z_90"
]



# # %%
import numpy as np

# Define the Pauli matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Define rotation matrices
def Rx(angle):
    return np.cos(angle/2) * I - 1j * np.sin(angle/2) * X

def Ry(angle):
    return np.cos(angle/2) * I - 1j * np.sin(angle/2) * Y

def Rz(angle):
    return np.array([[np.exp(-1j*angle/2), 0],
                     [0, np.exp(1j*angle/2)]])

# Define the quantum gates
X180 = Rx(np.pi)
X_180 = Rx(-np.pi)
Y180 = Ry(np.pi)
X90 = Rx(np.pi/2)
X_90 = Rx(-np.pi/2)
Y90 = Ry(np.pi/2)
Y_90 = Ry(-np.pi/2)
Z180 = Rz(np.pi)
Z90 = Rz(np.pi/2)
Z_90 = Rz(-np.pi/2)
Z_180 = Rz(-np.pi)

# Store the gates in a dictionary for easy access
gates = {
    "X180": X180,
    "X_180": X_180,
    "Y180": Y180,
    "X90": X90,
    "X_90": X_90,
    "Y90": Y90,
    "-Y90": Y_90,
    "Z180": Z180,
    "Z90": Z90,
    "Z_90": Z_90,
    "-Z180": Z_180,
    "I": I
}

# Multiply two gates
def multiply_gates(gate1_name, gate2_name):
    result = np.dot(gates[gate1_name], gates[gate2_name])
    return np.round(result, 2)  # Round to 2 decimal places


# %%
gates_list = [
    "I",	
    "X180",
    "Z_90 X180 Z90",
    "X90 Z180 Z90",
    
    "X90 Z_90 X90 Z90",
    "X90 Z90 X90 Z_90",
    "X_90 Z_90 X90 Z90",
    "X_90 Z90 X90 Z_90",
    "Z_90 X90 Z90 X90",
    "Z_90 X90 Z90 X_90",
    "Z90 X90 Z_90 X90",
    "Z90 X90 Z_90 X_90",
    
    "X90",
    "X_90",
    "Z_90 X90 Z90",
    "Z90 X90 Z_90",
    "Z90",
    "Z_90",
    
    "X90 Z_90 X90",
    "X90 Z90 X90",
    "Z_90 X90 Z90 X90 Z90",
    "Z_90 X90 Z90 X_90 Z_90",
    "X180 Z90",
    "X180 Z_90"
]

# Function to check if two gates are equivalent up to a phase factor
def are_equivalent_up_to_phase(gate1, gate2):
    # Calculate the ratio of corresponding elements
    ratios = gate1 / (gate2+ 1e-7)
    # Check if all ratios are equal (up to numerical precision)
    return np.allclose(ratios, np.eye(2), atol=1e-6)

# Dictionary to store the gates
gates_dict = {}

# Helper function to parse rotation angle
def parse_angle(rotation):
    rotation = rotation.lower()
    if rotation.startswith('-'):
        return -parse_angle(rotation[1:])
    elif rotation in ['x', 'y', 'z']:
        return 90
    elif rotation in ['x90', 'y90', 'z90']:
        return 90
    elif rotation in ['x180', 'y180', 'z180']:
        return 180
    elif rotation.startswith('x') or rotation.startswith('y') or rotation.startswith('z'):
        try:
            return float(rotation[1:])
        except ValueError:
            raise ValueError(f"Unexpected rotation angle: {rotation}")
    else:
        raise ValueError(f"Unexpected rotation angle: {rotation}")

# Calculate matrices for all gates and store them
for gate in gates_list:
    if " " not in gate:
        gates_dict[gate] = gates[gate]
    else:
        # For compound gates, split into individual gates
        subgates = gate.split()
        # Multiply the individual rotation matrices
        matrix = np.eye(2)
        for subgate in subgates:
            matrix = matrix @ gates[subgate]
        gates_dict[gate] = matrix

# %% 
# Multiply each element in gates_dict with all other elements
# and check which combinations are close to the identity matrix
identity = np.eye(2)
# Function to check if two gates are equivalent up to a phase factor
def are_equivalent_up_to_phase(gate1, gate2):
    matrixes_are_close = False
    for phase in np.linspace(0, 2*np.pi, 5):
        if np.allclose(gate1, gate2 * np.exp(1j*phase), atol=1e-10):
            matrixes_are_close = True
            break
    return matrixes_are_close

print("Combinations close to the identity matrix:")
identity_combinations = []
identity_combinations = []
for i, (gate1, matrix1) in enumerate(gates_dict.items()):
    for j, (gate2, matrix2) in enumerate(gates_dict.items()):
        result = matrix1 @ matrix2
        if are_equivalent_up_to_phase(result, identity):
            identity_combinations.append((i, j))
            print(f"{gate1} * {gate2}")


# You can also store these combinations if needed

print(f"\nTotal number of combinations close to identity: {len(identity_combinations)}")

# %%
multiplication_table = np.zeros((len(gates_dict), len(gates_dict)))
for i, (gate1, matrix1) in enumerate(gates_dict.items()):
    for j, (gate2, matrix2) in enumerate(gates_dict.items()):
        result = matrix1 @ matrix2
        for k, (gate3, matrix3) in enumerate(gates_dict.items()):
            if are_equivalent_up_to_phase(result, matrix3):
                multiplication_table[i, j] = int(k)

# Convert multiplication_table to an array of integers
multiplication_table = multiplication_table.astype(int)
# multiplication_table = multiplication_table.transpose()

print("Multiplication table:")
print(multiplication_table)

# %%
# print( np.round(gates["X180"] @ gates["Z90"], 2))
# print( np.round(gates["X180"] @ gates["Z_90"], 2))
print( np.round(1j*gates["X180"] @ gates["Z90"] @ gates["X180"] @ gates["Z_90"], 2))



# %%
gates_dict[0]
# %%
