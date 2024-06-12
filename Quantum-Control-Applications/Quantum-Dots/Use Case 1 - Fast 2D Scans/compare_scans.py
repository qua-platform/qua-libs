import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


N = 101  # Number of points in each direction
measurement_time = 1000  # in ns
font_position_correction = 0.06
tau_biasT = 200e3 * 100  # Bias-T time constant; *100 to have it in ns


def high_pass(data, tau):
    res = signal.butter(1, 1 / tau, btype="high", analog=False)
    return signal.lfilter(res[0], res[1], data)


##################################################
# Raster scan
##################################################
plt.figure("Raster Scan")

# Plot 2D map if N < 10
X = np.arange(0, N**2).reshape(N, N)
if N < 10:
    plt.subplot(121)
    plt.imshow(X.T, origin="lower")
    for i, values in enumerate(X):
        for j, num in enumerate(values):
            plt.text(i - font_position_correction, j - font_position_correction, num)

    plt.axis("off")

# Derive output voltages without and with high pass filtering
output_1 = np.full(shape=(N**2 + 2, measurement_time), fill_value=0.0)
output_2 = np.full(shape=(N**2 + 2, measurement_time), fill_value=0.0)

for i, values in enumerate(X):
    for j, num in enumerate(values):
        output_1[num + 1, :] = np.full(measurement_time, fill_value=(i - (N - 1) / 2) / (N - 1))
        output_2[num + 1, :] = np.full(measurement_time, fill_value=(j - (N - 1) / 2) / (N - 1))
if N < 10:
    plt.subplot(122)

plt.plot(output_1.flatten()[::100] + 1.1, "b", label="V_rg")
plt.plot(output_2.flatten()[::100], "r", label="V_lg")
output_1_filter = high_pass(output_1.flatten(), tau_biasT)
output_2_filter = high_pass(output_2.flatten(), tau_biasT)
plt.plot(output_1_filter.flatten()[::100] + 1.1, "c--", label="V_rg filtered")
plt.plot(output_2_filter.flatten()[::100], "m--", label="V_lg filtered")

plt.xlabel("time (ns)")
plt.ylabel("output voltage (V)")
plt.legend()
plt.show()

print(f"Averaged error per step: {np.average(np.abs(output_1.flatten()-output_1_filter)[:])*100:.1f} %")


##################################################
# Spiral scan
##################################################
def spiral(N: int):
    N = N if N % 2 == 1 else N + 1
    i, j = (N - 1) // 2, (N - 1) // 2
    order = np.zeros(shape=(N, N), dtype=int)

    sign = +1
    number_of_moves = 1
    total_moves = 0
    while total_moves < N**2 - N:
        for _ in range(number_of_moves):
            i = i + sign
            total_moves = total_moves + 1
            order[i, j] = total_moves

        for _ in range(number_of_moves):
            j = j + sign
            total_moves = total_moves + 1
            order[i, j] = total_moves
        sign = sign * -1
        number_of_moves = number_of_moves + 1

    for _ in range(number_of_moves - 1):
        i = i + sign
        total_moves = total_moves + 1
        order[i, j] = total_moves

    return order


plt.figure("Spiral Scan")

# Plot 2D map if N < 10
order = spiral(N)
if N < 10:
    plt.subplot(121)
    plt.imshow(order.T, origin="lower")
    for i, values in enumerate(order):
        for j, num in enumerate(values):
            plt.text(i - font_position_correction, j - font_position_correction, num)
    plt.axis("off")

# Derive output voltages without and with high pass filtering
output_1 = np.full(shape=(N**2 + 1, measurement_time), fill_value=0.0)
output_2 = np.full(shape=(N**2 + 1, measurement_time), fill_value=0.0)

for i, values in enumerate(order):
    for j, num in enumerate(values):
        output_1[num, :] = np.full(measurement_time, fill_value=(i - (N - 1) / 2) / (N - 1))
        output_2[num, :] = np.full(measurement_time, fill_value=(j - (N - 1) / 2) / (N - 1))

if N < 10:
    plt.subplot(122)
plt.plot(output_1.flatten()[::100] + 1.1, "b", label="V_rg")
plt.plot(output_2.flatten()[::100], "r", label="V_lg")
output_1_filter = high_pass(output_1.flatten(), tau_biasT)
output_2_filter = high_pass(output_2.flatten(), tau_biasT)
plt.plot(output_1_filter.flatten()[::100] + 1.1, "c--", label="V_rg filtered")
plt.plot(output_2_filter.flatten()[::100], "m--", label="V_lg filtered")
plt.xlabel("time (ns)")
plt.ylabel("output voltage (V)")
plt.legend()
plt.show()
print(f"Averaged error per step: {np.average(np.abs(output_1.flatten()-output_1_filter)[:])*100:.1f} %")


##################################################
# Diagonal scan
##################################################
def diag(N: int):
    order = np.zeros(shape=(N, N), dtype=int)
    label = np.zeros(shape=(N, N), dtype=int)
    sign_x = +1
    number_of_moves = 1
    total_moves = 0
    i, j = 0, 0

    order[i, j] = total_moves
    label[i, j] = str(total_moves)
    total_moves += 1

    shift_x = False
    while total_moves < N**2 // 2:
        if not shift_x:
            j += 1
        elif shift_x:
            i += 1

        shift_x = not shift_x
        order[i, j] = total_moves
        label[i, j] = str(total_moves)
        total_moves += 1

        for _ in range(number_of_moves):
            i = i + sign_x
            j = j - sign_x

            order[i, j] = total_moves
            label[i, j] = str(total_moves)
            total_moves = total_moves + 1
        sign_x = -sign_x
        number_of_moves = number_of_moves + 1

    if N % 2 == 1:
        shift_x = True
    else:
        shift_x = False
    number_of_moves = number_of_moves - 2
    while total_moves < N**2:
        if not shift_x:
            j += 1

        elif shift_x:
            i += 1
        shift_x = not shift_x
        order[i, j] = total_moves
        label[i, j] = str(total_moves)
        total_moves += 1

        for _ in range(number_of_moves):
            i = i + sign_x
            j = j - sign_x

            order[i, j] = total_moves
            label[i, j] = str(total_moves)
            total_moves = total_moves + 1
        sign_x = -sign_x
        number_of_moves = number_of_moves - 1

    return order, label


plt.figure("Diagonal Scan")

# Plot 2D map if N < 10
order, label = diag(N)
if N < 10:
    plt.subplot(121)
    plt.imshow(order.T, origin="lower")
    for i in range(N):
        for j in range(N):
            plt.text(i - font_position_correction, j - font_position_correction, label[i, j])
    plt.axis("off")

# Derive output voltages without and with high pass filtering
output_1 = np.full(shape=(N**2 + 1, measurement_time), fill_value=0.0)
output_2 = np.full(shape=(N**2 + 1, measurement_time), fill_value=0.0)

for i, values in enumerate(order):
    for j, num in enumerate(values):
        output_1[num, :] = np.full(measurement_time, fill_value=(i - (N - 1) / 2) / (N - 1))
        output_2[num, :] = np.full(measurement_time, fill_value=(j - (N - 1) / 2) / (N - 1))
if N < 10:
    plt.subplot(122)
plt.plot(output_1.flatten()[::100] + 1.1, "b", label="V_rg")
plt.plot(output_2.flatten()[::100], "r", label="V_lg")
output_1_filter = high_pass(output_1.flatten(), tau_biasT)
output_2_filter = high_pass(output_2.flatten(), tau_biasT)
plt.plot(output_1_filter.flatten()[::100] + 1.1, "c--", label="V_rg filtered")
plt.plot(output_2_filter.flatten()[::100], "m--", label="V_lg filtered")
plt.xlabel("time (ns)")
plt.ylabel("output voltage (V)")
plt.legend()
plt.show()

print(f"Averaged error per step: {np.average(np.abs(output_1.flatten()-output_1_filter)[:])*100:.1f} %")
