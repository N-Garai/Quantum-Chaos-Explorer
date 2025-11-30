"""
Quantum Chaos Engine - Core Physics Module (FIXED VERSION)
===========================================================
This module implements the quantum mechanics behind fractal generation.
It maps classical text input to quantum circuits and computes probability
distributions across 2D coordinate spaces.

BUG FIX: All fractal functions now handle NumPy types correctly

Author: N-Garai
Date: November 2025
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Tuple, List, Callable
import hashlib
import re


def hash_text_to_circuit(seed_text: str, num_qubits: int = 1) -> QuantumCircuit:
    """
    Converts a text string into a unique quantum circuit using character-to-gate mapping.

    Physics Explanation:
    --------------------
    Each character creates quantum interference patterns through:
    - Even ASCII → Hadamard (H): Creates superposition, splits |0⟩ into |0⟩+|1⟩
    - Odd ASCII → Rotation (Ry): Rotates qubit on Bloch sphere by angle θ = ASCII/10
    - Vowels → T gate: Adds π/4 phase shift (introduces quantum "twist")

    Parameters:
    -----------
    seed_text : str
        Input string that defines the quantum circuit structure
    num_qubits : int
        Number of qubits in the circuit (default: 1)

    Returns:
    --------
    QuantumCircuit : A Qiskit quantum circuit object
    """
    qc = QuantumCircuit(num_qubits)
    vowels = set('aeiouAEIOU')

    # Initialize circuit with seed-dependent starting state
    hash_val = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % 100
    qc.ry(hash_val / 10, 0)  # Initial rotation based on text hash

    for char in seed_text:
        ascii_val = ord(char)

        # Gate selection based on character properties
        if ascii_val % 2 == 0:
            # Even ASCII: Apply Hadamard for maximum superposition
            qc.h(0)
        else:
            # Odd ASCII: Apply controlled rotation (creates spiral patterns)
            angle = (ascii_val % 100) / 10.0
            qc.ry(angle, 0)

        # Vowels add phase complexity (T gate = e^(iπ/4) phase)
        if char in vowels:
            qc.t(0)

        # Add small Z rotation for character uniqueness
        qc.rz((ascii_val % 50) / 25.0, 0)

    return qc


def parse_custom_equation(equation: str) -> Callable:
    """
    Parses a user-provided mathematical equation for fractal generation.

    Supported variables: x, y, z (from quantum state)
    Supported functions: sin, cos, tan, exp, log, sqrt, abs

    Example equations:
    - "sin(x) + cos(y)"
    - "x**2 + y**2 - z"
    - "exp(-x**2 - y**2)"

    Returns:
    --------
    Callable : A function that takes (x, y, z) and returns a scalar value
    """
    # Sanitize and prepare the equation
    equation = equation.lower().strip()

    # Create safe namespace with math functions
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }

    def equation_func(x, y, z):
        """Evaluates the user equation at given coordinates."""
        try:
            # FIX: Convert NumPy types to Python types
            x = float(x)
            y = float(y)
            z = float(z)

            # Add coordinates to namespace
            local_dict = safe_dict.copy()
            local_dict.update({'x': x, 'y': y, 'z': z})

            # Evaluate the equation
            result = eval(equation, {"__builtins__": {}}, local_dict)
            return result
        except Exception as e:
            # Fallback to default if equation fails
            return x**2 + y**2

    return equation_func


def get_predefined_fractal(fractal_name: str) -> Callable:
    """
    Returns a predefined fractal equation function.

    Available fractals:
    - mandelbrot: Classic Mandelbrot set
    - julia: Julia set
    - burning_ship: Burning Ship fractal
    - newton: Newton fractal
    - phoenix: Phoenix fractal
    """
    fractals = {
        'mandelbrot': lambda x, y, z: mandelbrot_iteration(x, y),
        'julia': lambda x, y, z: julia_iteration(x, y, -0.7, 0.27015),
        'burning_ship': lambda x, y, z: burning_ship_iteration(x, y),
        'newton': lambda x, y, z: newton_iteration(x, y),
        'phoenix': lambda x, y, z: phoenix_iteration(x, y),
        'quantum_mandelbrot': lambda x, y, z: mandelbrot_iteration(x * z, y * (1 - z)),
        'heart': lambda x, y, z: heart_shape(x, y),
        'spiral': lambda x, y, z: spiral_pattern(x, y, z),
    }

    return fractals.get(fractal_name.lower(), fractals['mandelbrot'])


def mandelbrot_iteration(x, y, max_iter=50):
    """Mandelbrot set iteration - FIXED: Handle NumPy types"""
    # FIX: Convert NumPy types to Python native types
    x = float(x)
    y = float(y)
    max_iter = int(max_iter)

    c = complex(x, y)
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n / max_iter
        z = z*z + c
    return 1.0


def julia_iteration(x, y, c_real=-0.7, c_imag=0.27015, max_iter=50):
    """Julia set iteration - FIXED"""
    x = float(x)
    y = float(y)
    max_iter = int(max_iter)

    c = complex(c_real, c_imag)
    z = complex(x, y)
    for n in range(max_iter):
        if abs(z) > 2:
            return n / max_iter
        z = z*z + c
    return 1.0


def burning_ship_iteration(x, y, max_iter=50):
    """Burning Ship fractal iteration - FIXED"""
    x = float(x)
    y = float(y)
    max_iter = int(max_iter)

    c = complex(x, y)
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n / max_iter
        z = complex(abs(z.real), abs(z.imag))**2 + c
    return 1.0


def newton_iteration(x, y, max_iter=30):
    """Newton fractal (roots of z^3 - 1) - FIXED"""
    x = float(x)
    y = float(y)
    max_iter = int(max_iter)

    z = complex(x, y)
    for n in range(max_iter):
        if abs(z) < 0.001:
            return n / max_iter
        z = z - (z**3 - 1) / (3 * z**2 + 1e-10)
    return np.abs(z.real % 1)


def phoenix_iteration(x, y, max_iter=50):
    """Phoenix fractal - FIXED"""
    x = float(x)
    y = float(y)
    max_iter = int(max_iter)

    c = complex(0.56667, -0.5)
    z = complex(x, y)
    z_prev = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n / max_iter
        z_new = z**2 + c.real + c.imag * z_prev
        z_prev = z
        z = z_new
    return 1.0


def heart_shape(x, y):
    """Heart-shaped equation - FIXED"""
    x = float(x)
    y = float(y)
    return (x**2 + y**2 - 1)**3 - x**2 * y**3


def spiral_pattern(x, y, z):
    """Spiral pattern using polar coordinates - FIXED"""
    x = float(x)
    y = float(y)
    z = float(z)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.sin(5 * theta + 10 * r * z)


def generate_fractal_grid(
    resolution: int,
    zoom: float,
    center_x: float,
    center_y: float,
    seed_text: str,
    fractal_mode: str = "predefined",
    fractal_name: str = "mandelbrot",
    custom_equation: str = None
) -> Tuple[np.ndarray, QuantumCircuit]:
    """
    Generates a 2D quantum-classical hybrid fractal - FIXED: Handle NumPy types

    NEW FEATURE: Combines quantum circuits with classical fractal mathematics.

    Parameters:
    -----------
    fractal_mode : str
        "predefined" - Use built-in fractals
        "custom" - Use user-provided equation
        "quantum_only" - Pure quantum interference
    fractal_name : str
        Name of predefined fractal (if mode is "predefined")
    custom_equation : str
        Custom mathematical equation (if mode is "custom")
    """
    # Generate quantum circuit from seed text
    base_circuit = hash_text_to_circuit(seed_text)

    # Create coordinate meshgrid
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Initialize result grid
    Z = np.zeros((resolution, resolution))

    # Determine fractal function
    if fractal_mode == "custom" and custom_equation:
        fractal_func = parse_custom_equation(custom_equation)
    elif fractal_mode == "predefined":
        fractal_func = get_predefined_fractal(fractal_name)
    else:
        fractal_func = None  # Pure quantum mode

    # Generate fractal with quantum modulation
    for i in range(resolution):
        for j in range(resolution):
            # Get quantum probability for this pixel
            qc_pixel = base_circuit.copy()

            # Map pixel coordinates to quantum angles
            # FIX: Convert to Python float
            phi = float(np.pi * X[i, j])
            theta = float(np.pi * (Y[i, j] + 1) / 2)

            qc_pixel.ry(theta, 0)
            qc_pixel.rz(phi, 0)

            # Compute quantum state
            state = Statevector.from_instruction(qc_pixel)
            probs = state.probabilities()
            quantum_value = float(probs[0])  # FIX: Convert to Python float

            # Combine quantum with classical fractal
            if fractal_func:
                # FIX: Pass Python floats to fractal function
                classical_value = fractal_func(float(X[i, j]), float(Y[i, j]), quantum_value)
                # Hybrid: modulate fractal with quantum probability
                Z[i, j] = float(classical_value) * quantum_value
            else:
                # Pure quantum mode
                Z[i, j] = probs[0] - probs[1]

    # Normalize to [0, 1]
    Z = np.nan_to_num(Z)  # Handle any NaN values
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-10)

    return Z, base_circuit


def get_circuit_info(circuit: QuantumCircuit) -> dict:
    """Extract metadata from quantum circuit."""
    gate_counts = {}
    for instruction in circuit.data:
        gate_name = instruction[0].name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    return {
        'depth': circuit.depth(),
        'num_gates': len(circuit.data),
        'gate_counts': gate_counts,
        'num_qubits': circuit.num_qubits
    }


# List of available predefined fractals for UI
PREDEFINED_FRACTALS = [
    "Mandelbrot",
    "Julia",
    "Burning Ship",
    "Newton",
    "Phoenix",
    "Quantum Mandelbrot",
    "Heart",
    "Spiral"
]
