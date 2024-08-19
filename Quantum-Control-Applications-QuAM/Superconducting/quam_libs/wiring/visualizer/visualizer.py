import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the chassis dimensions
CHASSIS_WIDTH = 8
CHASSIS_HEIGHT = 3  # Updated height for chassis

# Define the port positions in a FEM (normalized within a slot)
PORT_SPACING_FACTOR = 0.1  # Close spacing between ports
PORT_POSITIONS = {
    'output': [(0.25, 1 - i * PORT_SPACING_FACTOR) for i in range(8)],
    'input': [(0.75, 1 - i * PORT_SPACING_FACTOR * 4) for i in range(2)]
}

# Define a function to draw and annotate ports
def annotate_port(ax, port, annotations, assigned, color):
    pos = PORT_POSITIONS[port.io_type][port.port - 1]
    x = port.slot - 1 + pos[0]
    y = pos[1] * CHASSIS_HEIGHT - 0.5

    # Draw port as filled or empty circle based on assignment
    fill_color = color if assigned else 'none'
    ax.add_patch(patches.Circle((x, y), 0.1, edgecolor='black', facecolor=fill_color))

    # Place grouped annotations to the left of the port with a semi-transparent white background
    for i, annotation in enumerate(annotations):
        ax.text(x - 0.15, y + 0.15 * i, annotation, ha='right', va='center', fontsize=14, color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# Define a function to label the slots
def label_slot(ax, slot, fem_type):
    if fem_type:
        label = 'MW-FEM' if fem_type == 'mw' else 'LF-FEM'
        ax.text(slot - 0.5, CHASSIS_HEIGHT + 0.2, label, ha='center', va='bottom', fontsize=12, fontweight='bold')

# Define a function to get port color based on line type
def get_color_for_line_type(line_type):
    color_map = {
        'rr': 'blue',
        'xy': 'green',
        'z': 'orange',
        'c': 'purple'
    }
    return color_map.get(line_type, 'grey')

# Define a function to visualize the chassis and FEMs
def visualize_chassis(qubit_dict):
    fig, ax = plt.subplots(figsize=(CHASSIS_WIDTH, CHASSIS_HEIGHT))

    # Draw the chassis slots with boundary lines
    for slot in range(1, CHASSIS_WIDTH + 1):
        # Set light grey background for the FEMs
        ax.add_patch(patches.Rectangle((slot - 1, 0), 1, CHASSIS_HEIGHT, facecolor='lightgrey', edgecolor='black'))

    # Track the FEM types per slot and ports assignments
    fem_types = {slot: None for slot in range(1, CHASSIS_WIDTH + 1)}
    port_assignments = {slot: {'input': [False] * 2, 'output': [False] * 8} for slot in range(1, CHASSIS_WIDTH + 1)}
    annotations_map = {slot: {'input': [[] for _ in range(2)], 'output': [[] for _ in range(8)]} for slot in range(1, CHASSIS_WIDTH + 1)}

    # Annotate the ports and determine the FEM types
    for qubit_ref, element in qubit_dict.items():
        for channel_type, channels in element.channels.items():
            for port in channels:
                annotation = f"q{qubit_ref.index if hasattr(qubit_ref, 'index') else f'{qubit_ref.control_index}{qubit_ref.target_index}'}.{channel_type}"
                annotations_map[port.slot][port.io_type][port.port - 1].append(annotation)
                port_assignments[port.slot][port.io_type][port.port - 1] = True
                if 'mw' in type(port).__name__.lower():
                    fem_types[port.slot] = 'mw'
                elif 'lf' in type(port).__name__.lower():
                    fem_types[port.slot] = 'lf'

    # Draw ports with annotations
    for slot in range(1, CHASSIS_WIDTH + 1):
        for io_type, ports in port_assignments[slot].items():
            has_assigned_ports = any(assigned for assigned in ports)
            for i, assigned in enumerate(ports):
                port = type('Port', (object,), {'slot': slot, 'port': i + 1, 'io_type': io_type})()
                if assigned:
                    line_type = annotations_map[slot][io_type][i][0].split('.')[1] if annotations_map[slot][io_type][i] else 'rr'
                    color = get_color_for_line_type(line_type)
                    annotate_port(ax, port, annotations_map[slot][io_type][i], assigned=True, color=color)
                elif has_assigned_ports:
                    annotate_port(ax, port, [], assigned=False, color='none')
        label_slot(ax, slot, fem_types[slot])

    # Draw slot boundaries explicitly
    for slot in range(CHASSIS_WIDTH):
        ax.plot([slot, slot], [0, CHASSIS_HEIGHT], color='black', lw=1)

    # Set the limits and display the plot
    ax.set_xlim(0, CHASSIS_WIDTH)
    ax.set_ylim(0, CHASSIS_HEIGHT)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()
