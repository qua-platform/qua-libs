import numpy as np

c1_table_XZ = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23],
       [ 1,  0,  3,  2,  6,  7,  4,  5, 11, 10,  9,  8, 13, 12, 18, 19,
        22, 23, 14, 15, 21, 20, 16, 17],
       [ 2,  3,  0,  1,  7,  6,  5,  4, 10, 11,  8,  9, 20, 21, 15, 14,
        23, 22, 19, 18, 12, 13, 17, 16],
       [ 3,  2,  1,  0,  5,  4,  7,  6,  9,  8, 11, 10, 21, 20, 19, 18,
        17, 16, 15, 14, 13, 12, 23, 22],
       [ 4,  7,  5,  6, 11,  8,  9, 10,  2,  3,  1,  0, 22, 17, 21, 12,
        14, 18, 13, 20, 23, 16, 15, 19],
       [ 5,  6,  4,  7, 10,  9,  8, 11,  1,  0,  2,  3, 23, 16, 12, 21,
        19, 15, 20, 13, 22, 17, 18, 14],
       [ 6,  5,  7,  4,  8, 11, 10,  9,  3,  2,  0,  1, 16, 23, 20, 13,
        18, 14, 12, 21, 17, 22, 19, 15],
       [ 7,  4,  6,  5,  9, 10, 11,  8,  0,  1,  3,  2, 17, 22, 13, 20,
        15, 19, 21, 12, 16, 23, 14, 18],
       [ 8,  9, 11, 10,  1,  3,  2,  0,  7,  4,  5,  6, 19, 14, 22, 16,
        20, 12, 23, 17, 15, 18, 13, 21],
       [ 9,  8, 10, 11,  2,  0,  1,  3,  6,  5,  4,  7, 14, 19, 23, 17,
        13, 21, 22, 16, 18, 15, 20, 12],
       [10, 11,  9,  8,  3,  1,  0,  2,  4,  7,  6,  5, 18, 15, 17, 23,
        12, 20, 16, 22, 14, 19, 21, 13],
       [11, 10,  8,  9,  0,  2,  3,  1,  5,  6,  7,  4, 15, 18, 16, 22,
        21, 13, 17, 23, 19, 14, 12, 20],
       [12, 13, 21, 20, 18, 19, 14, 15, 22, 17, 23, 16,  1,  0,  4,  5,
         8, 10,  6,  7,  2,  3, 11,  9],
       [13, 12, 20, 21, 14, 15, 18, 19, 16, 23, 17, 22,  0,  1,  6,  7,
        11,  9,  4,  5,  3,  2,  8, 10],
       [14, 19, 15, 18, 22, 16, 23, 17, 20, 21, 12, 13,  8,  9,  2,  0,
         6,  4,  1,  3, 10, 11,  7,  5],
       [15, 18, 14, 19, 17, 23, 16, 22, 12, 13, 20, 21, 10, 11,  0,  2,
         5,  7,  3,  1,  8,  9,  4,  6],
       [16, 23, 22, 17, 12, 21, 20, 13, 19, 14, 15, 18,  5,  6,  8, 11,
         3,  0, 10,  9,  7,  4,  1,  2],
       [17, 22, 23, 16, 21, 12, 13, 20, 14, 19, 18, 15,  4,  7,  9, 10,
         0,  3, 11,  8,  6,  5,  2,  1],
       [18, 15, 19, 14, 16, 22, 17, 23, 21, 20, 13, 12, 11, 10,  3,  1,
         4,  6,  0,  2,  9,  8,  5,  7],
       [19, 14, 18, 15, 23, 17, 22, 16, 13, 12, 21, 20,  9,  8,  1,  3,
         7,  5,  2,  0, 11, 10,  6,  4],
       [20, 21, 13, 12, 19, 18, 15, 14, 17, 22, 16, 23,  3,  2,  7,  6,
        10,  8,  5,  4,  0,  1,  9, 11],
       [21, 20, 12, 13, 15, 14, 19, 18, 23, 16, 22, 17,  2,  3,  5,  4,
         9, 11,  7,  6,  1,  0, 10,  8],
       [22, 17, 16, 23, 13, 20, 21, 12, 15, 18, 19, 14,  7,  4, 11,  8,
         2,  1,  9, 10,  5,  6,  0,  3],
       [23, 16, 17, 22, 20, 13, 12, 21, 18, 15, 14, 19,  6,  5, 10,  9,
         1,  2,  8, 11,  4,  7,  3,  0]])

# # %%
# import numpy as np

# gates_list = [
#     "I",	
#     "X180",
#     "Z-90 X180 Z90",
#     "X90 Z180 X90",
    
#     "X90 Z_90 X90 Z90",
#     "X90 Z90 X90 Z_90",
#     "X_90 Z_90 X90 Z90",
#     "X_90 Z90 X90 Z_90",
#     "Z_90 X90 Z90 X90",
#     "Z_90 X90 Z90 X_90",
#     "Z90 X90 Z_90 X90",
#     "Z90 X90 Z_90 X_90",
    
#     "X90",
#     "X-90",
#     "Z_90 X90 Z90",
#     "Z90 X90 Z_90",
#     "Z90",
#     "Z-90",
    
#     "X90 Z_90 X90",
#     "X90 Z90 X90",
#     "Z_90 X90 Z90 X90 Z90",
#     "Z_90 X90 Z90 X_90 Z_90",
#     "X180 Z90",
#     "X180 Z_90"
# ]



# # # %%
# import numpy as np

# # Define the Pauli matrices
# I = np.array([[1, 0], [0, 1]])
# X = np.array([[0, 1], [1, 0]])
# Y = np.array([[0, -1j], [1j, 0]])
# Z = np.array([[1, 0], [0, -1]])

# # Define rotation matrices
# def Rx(angle):
#     return np.cos(angle/2) * I - 1j * np.sin(angle/2) * X

# def Ry(angle):
#     return np.cos(angle/2) * I - 1j * np.sin(angle/2) * Y

# def Rz(angle):
#     return np.array([[np.exp(-1j*angle/2), 0],
#                      [0, np.exp(1j*angle/2)]])

# # Define the quantum gates
# X180 = Rx(np.pi)
# X_180 = Rx(-np.pi)
# Y180 = Ry(np.pi)
# X90 = Rx(np.pi/2)
# X_90 = Rx(-np.pi/2)
# Y90 = Ry(np.pi/2)
# Y_90 = Ry(-np.pi/2)
# Z180 = Rz(np.pi)
# Z90 = Rz(np.pi/2)
# Z_90 = Rz(-np.pi/2)
# Z_180 = Rz(-np.pi)

# # Store the gates in a dictionary for easy access
# gates = {
#     "X180": X180,
#     "X_180": X_180,
#     "Y180": Y180,
#     "X90": X90,
#     "X_90": X_90,
#     "Y90": Y90,
#     "-Y90": Y_90,
#     "Z180": Z180,
#     "Z90": Z90,
#     "Z_90": Z_90,
#     "-Z180": Z_180,
#     "I": I
# }

# # Multiply two gates
# def multiply_gates(gate1_name, gate2_name):
#     result = np.dot(gates[gate1_name], gates[gate2_name])
#     return np.round(result, 2)  # Round to 2 decimal places


# # %%
# gates_list = [
#     "I",	
#     "X180",
#     "Z_90 X180 Z90",
#     "X90 Z180 X90",
    
#     "X90 Z_90 X90 Z90",
#     "X90 Z90 X90 Z_90",
#     "X_90 Z_90 X90 Z90",
#     "X_90 Z90 X90 Z_90",
#     "Z_90 X90 Z90 X90",
#     "Z_90 X90 Z90 X_90",
#     "Z90 X90 Z_90 X90",
#     "Z90 X90 Z_90 X_90",
    
#     "X90",
#     "X_90",
#     "Z_90 X90 Z90",
#     "Z90 X90 Z_90",
#     "Z90",
#     "Z_90",
    
#     "X90 Z_90 X90",
#     "X90 Z90 X90",
#     "Z_90 X90 Z90 X90 Z90",
#     "Z_90 X90 Z90 X_90 Z_90",
#     "X180 Z90",
#     "X180 Z_90"
# ]

# # Function to check if two gates are equivalent up to a phase factor
# def are_equivalent_up_to_phase(gate1, gate2):
#     # Calculate the ratio of corresponding elements
#     ratios = gate1 / (gate2+ 1e-7)
#     # Check if all ratios are equal (up to numerical precision)
#     return np.allclose(ratios, np.eye(2), atol=1e-6)

# # Dictionary to store the gates
# gates_dict = {}

# # Helper function to parse rotation angle
# def parse_angle(rotation):
#     rotation = rotation.lower()
#     if rotation.startswith('-'):
#         return -parse_angle(rotation[1:])
#     elif rotation in ['x', 'y', 'z']:
#         return 90
#     elif rotation in ['x90', 'y90', 'z90']:
#         return 90
#     elif rotation in ['x180', 'y180', 'z180']:
#         return 180
#     elif rotation.startswith('x') or rotation.startswith('y') or rotation.startswith('z'):
#         try:
#             return float(rotation[1:])
#         except ValueError:
#             raise ValueError(f"Unexpected rotation angle: {rotation}")
#     else:
#         raise ValueError(f"Unexpected rotation angle: {rotation}")

# # Calculate matrices for all gates and store them
# for gate in gates_list:
#     if " " not in gate:
#         gates_dict[gate] = gates[gate]
#     else:
#         # For compound gates, split into individual gates
#         subgates = gate.split()
#         # Multiply the individual rotation matrices
#         matrix = np.eye(2)
#         for subgate in subgates:
#             matrix = matrix @ gates[subgate]
#         gates_dict[gate] = matrix

# # %% 
# # Multiply each element in gates_dict with all other elements
# # and check which combinations are close to the identity matrix
# identity = np.eye(2)
# # Function to check if two gates are equivalent up to a phase factor
# def are_equivalent_up_to_phase(gate1, gate2):
#     matrixes_are_close = False
#     for phase in np.linspace(0, 2*np.pi, 5):
#         if np.allclose(gate1, gate2 * np.exp(1j*phase), atol=1e-2):
#             matrixes_are_close = True
#             break
#     return matrixes_are_close

# print("Combinations close to the identity matrix:")
# identity_combinations = []
# identity_combinations = []
# for i, (gate1, matrix1) in enumerate(gates_dict.items()):
#     for j, (gate2, matrix2) in enumerate(gates_dict.items()):
#         result = matrix1 @ matrix2
#         if are_equivalent_up_to_phase(result, identity):
#             identity_combinations.append((i, j))
#             print(f"{gate1} * {gate2}")


# # You can also store these combinations if needed

# print(f"\nTotal number of combinations close to identity: {len(identity_combinations)}")

# # %%
# multiplication_table = np.zeros((len(gates_dict), len(gates_dict))) - 1
# for i, (gate1, matrix1) in enumerate(gates_dict.items()):
#     for j, (gate2, matrix2) in enumerate(gates_dict.items()):
#         result = matrix1 @ matrix2
#         for k, (gate3, matrix3) in enumerate(gates_dict.items()):
#             if are_equivalent_up_to_phase(result, matrix3):
#                 multiplication_table[i, j] = int(k)

# # Convert multiplication_table to an array of integers
# multiplication_table = multiplication_table.astype(int)
# # multiplication_table = multiplication_table.transpose()

# print("Multiplication table:")
# print(multiplication_table)

# # %%
# multiplication_table
# # %%
