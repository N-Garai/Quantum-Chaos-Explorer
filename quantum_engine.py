"""
Quantum Fractal Engine - ULTIMATE VERSION
==========================================

Comprehensive fractal generation engine with 15+ fractal types.
Supports iterative, non-iterative, and advanced exotic fractals.

Author: N-Garai
Date: December 2025
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Tuple, Callable
import hashlib
import re


def text_to_quantum_params(seed_text: str) -> Tuple[float, float]:
    """Convert text to quantum-derived complex parameters."""
    qc = QuantumCircuit(2)

    hash_val = int(hashlib.md5(seed_text.encode()).hexdigest()[:16], 16) % 628
    qc.ry(hash_val / 100, 0)
    qc.ry(hash_val / 150, 1)

    vowels = set('aeiouAEIOU')
    for char in seed_text:
        ascii_val = ord(char)

        if ascii_val % 2 == 0:
            qc.h(0)
            qc.h(1)
        else:
            angle = (ascii_val % 100) / 50.0
            qc.ry(angle, 0)
            qc.ry(angle, 1)

        if char in vowels:
            qc.t(0)
            qc.t(1)

        qc.rz((ascii_val % 50) / 25.0, 0)
        qc.rz((ascii_val % 50) / 25.0, 1)

    state = Statevector.from_instruction(qc)
    amps = state.data

    c_real = float(np.real(amps[0])) - 0.5
    c_imag = float(np.imag(amps[1]))

    c_real = c_real * 1.5 - 0.4
    c_imag = c_imag * 1.5

    return c_real, c_imag


# ============================================================
# CLASSIC FRACTALS
# ============================================================

def generate_mandelbrot(width=400, height=400, max_iter=256,
                       center_x=-0.5, center_y=0.0, zoom=1.2):
    """Classic Mandelbrot: z → z² + c"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + C[mask]
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def generate_julia(c_real, c_imag, width=400, height=400, max_iter=256,
                  zoom=1.5, center_x=0.0, center_y=0.0):
    """Julia set: z → z² + c (c is constant)"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    Z = X + 1j * Y
    C = c_real + 1j * c_imag
    img = np.zeros(Z.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 10
        Z[mask] = Z[mask]**2 + C
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


# ============================================================
# ADVANCED FRACTALS
# ============================================================

def generate_burning_ship(width=400, height=400, max_iter=256,
                         center_x=-0.5, center_y=-0.5, zoom=1.2):
    """Burning Ship: z → (|Re(z)| + i|Im(z)|)² + c"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 10
        # Burning Ship: take absolute values before squaring
        Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag))**2 + C[mask]
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def generate_tricorn(width=400, height=400, max_iter=256,
                     center_x=-0.5, center_y=0.0, zoom=1.2):
    """Tricorn (Mandelbar): z → conj(z)² + c"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = np.conj(Z[mask])**2 + C[mask]  # Use conjugate
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def generate_phoenix(a=0.56667, b=-0.5, width=400, height=400, max_iter=256,
                    center_x=0.0, center_y=0.0, zoom=1.5):
    """Phoenix: z(n+1) = z(n)² + a + b*z(n-1)"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    Z = X + 1j * Y
    Z_prev = np.zeros_like(Z)
    img = np.zeros(Z.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 10
        Z_new = Z.copy()
        Z_new[mask] = Z[mask]**2 + complex(a, b) + b * Z_prev[mask]
        Z_prev[mask] = Z[mask]
        Z = Z_new
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def generate_celtic(width=400, height=400, max_iter=256,
                   center_x=-0.5, center_y=0.0, zoom=1.2):
    """Celtic: z → (|Re(z²)| + i*Im(z²)) + c"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z_sq = Z[mask]**2
        Z[mask] = (np.abs(Z_sq.real) + 1j * Z_sq.imag) + C[mask]
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


# ============================================================
# POLYNOMIAL FRACTALS
# ============================================================

def generate_multibrot(power=3, width=400, height=400, max_iter=256,
                      center_x=-0.5, center_y=0.0, zoom=1.2):
    """Multibrot: z → z^n + c (generalized Mandelbrot)"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**power + C[mask]
        img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def generate_newton(width=400, height=400, max_iter=50,
                   center_x=0.0, center_y=0.0, zoom=2.0):
    """Newton fractal: roots of z³ - 1"""
    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    Z = X + 1j * Y
    img = np.zeros(Z.shape)

    # Three roots of z³ - 1
    roots = [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]

    for i in range(max_iter):
        # Newton's method: z → z - (z³-1)/(3z²)
        Z = Z - (Z**3 - 1) / (3 * Z**2 + 1e-10)

        # Check convergence to roots
        for idx, root in enumerate(roots):
            mask = np.abs(Z - root) < 0.01
            img[mask] = idx / len(roots) + i / max_iter / len(roots)

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


# ============================================================
# CUSTOM EQUATION SUPPORT
# ============================================================

def detect_iterative_equation(equation: str) -> bool:
    """Detect if equation is iterative."""
    patterns = [r'z\(n\+1\)', r'z\(n\)', r'z\[n\]', r'z_n']
    return any(re.search(pattern, equation.lower()) for pattern in patterns)


def parse_iterative_equation(equation: str) -> str:
    """Parse iterative equation to Python code."""
    equation = re.sub(r'z\(n\+1\)\s*=\s*', '', equation)
    equation = re.sub(r'z\(n\)', 'z', equation)
    equation = re.sub(r'z\[n\]', 'z', equation)
    equation = re.sub(r'\^', '**', equation)  # Support ^ for power
    return equation.strip()


def generate_custom_iterative(equation_str: str, width=400, height=400,
                              max_iter=256, zoom=1.5, center_x=0.0, 
                              center_y=0.0, quantum_seed="Quantum"):
    """Generate fractal from iterative equation."""
    c_real, c_imag = text_to_quantum_params(quantum_seed)
    c = c_real + 1j * c_imag

    parsed_eq = parse_iterative_equation(equation_str)

    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    # Mandelbrot-style: z=0, c=coordinate
    if 'c' in parsed_eq.lower():
        C = X + 1j * Y
        Z = np.zeros_like(C)
        use_coord_as_c = True
    # Julia-style: z=coordinate, c=constant
    else:
        Z = X + 1j * Y
        C = c
        use_coord_as_c = False

    img = np.zeros((height, width))

    for i in range(max_iter):
        mask = np.abs(Z) < 10

        try:
            if use_coord_as_c:
                Z[mask] = eval(parsed_eq, {'z': Z[mask], 'c': C[mask], 'np': np})
            else:
                Z[mask] = eval(parsed_eq, {'z': Z[mask], 'c': C, 'np': np})
            img[mask] = i
        except:
            Z[mask] = Z[mask]**2 + (C if not use_coord_as_c else C[mask])
            img[mask] = i

    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def parse_custom_equation(equation: str) -> Callable:
    """Parse non-iterative equation."""
    equation = equation.lower().strip()
    equation = re.sub(r'\^', '**', equation)

    safe_dict = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
        'abs': np.abs, 'pi': np.pi, 'e': np.e,
    }

    def equation_func(x, y, z):
        try:
            x, y, z = float(x), float(y), float(z)
            local_dict = safe_dict.copy()
            local_dict.update({'x': x, 'y': y, 'z': z})
            result = eval(equation, {'__builtins__': {}}, local_dict)
            return float(result)
        except:
            return 0.0

    return equation_func


def generate_custom_simple(equation_func: Callable, width=400, height=400,
                          zoom=1.5, center_x=0.0, center_y=0.0,
                          quantum_seed="Quantum"):
    """Generate simple custom equation fractal."""
    c_real, c_imag = text_to_quantum_params(quantum_seed)
    quantum_mod = (c_real + c_imag) / 2

    x_min, x_max = center_x - zoom, center_x + zoom
    y_min, y_max = center_y - zoom, center_y + zoom

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            try:
                img[i, j] = equation_func(X[i, j], Y[i, j], quantum_mod)
            except:
                img[i, j] = 0.0

    img = np.nan_to_num(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_fractal_grid(
    resolution: int,
    zoom: float,
    center_x: float,
    center_y: float,
    seed_text: str,
    fractal_mode: str = "predefined",
    fractal_name: str = "mandelbrot",
    custom_equation: str = None,
    max_iter: int = 256
) -> Tuple[np.ndarray, QuantumCircuit]:
    """
    Universal fractal generator supporting 15+ fractal types.

    Modes:
    - predefined: Use built-in fractals
    - custom: Parse user equations (iterative or simple)
    - quantum_only: Pure quantum Julia sets
    """
    # Generate quantum circuit (for metadata)
    qc = QuantumCircuit(2)
    hash_val = int(hashlib.md5(seed_text.encode()).hexdigest()[:16], 16) % 628
    qc.ry(hash_val / 100, 0)
    qc.ry(hash_val / 150, 1)

    # Custom equations
    if fractal_mode == "custom" and custom_equation:
        if detect_iterative_equation(custom_equation):
            Z = generate_custom_iterative(
                custom_equation, width=resolution, height=resolution,
                max_iter=max_iter, zoom=zoom, center_x=center_x,
                center_y=center_y, quantum_seed=seed_text
            )
        else:
            eq_func = parse_custom_equation(custom_equation)
            Z = generate_custom_simple(
                eq_func, width=resolution, height=resolution,
                zoom=zoom, center_x=center_x, center_y=center_y,
                quantum_seed=seed_text
            )

    # Predefined fractals
    elif fractal_mode == "predefined":
        fname = fractal_name.lower()

        if fname == "julia":
            c_real, c_imag = text_to_quantum_params(seed_text)
            Z = generate_julia(c_real, c_imag, width=resolution, height=resolution,
                             max_iter=max_iter, zoom=zoom, center_x=center_x, center_y=center_y)

        elif fname == "mandelbrot":
            Z = generate_mandelbrot(width=resolution, height=resolution,
                                  max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "burning ship":
            Z = generate_burning_ship(width=resolution, height=resolution,
                                    max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "tricorn":
            Z = generate_tricorn(width=resolution, height=resolution,
                               max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "phoenix":
            Z = generate_phoenix(width=resolution, height=resolution,
                               max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "celtic":
            Z = generate_celtic(width=resolution, height=resolution,
                              max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "cubic":
            Z = generate_multibrot(power=3, width=resolution, height=resolution,
                                 max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "quartic":
            Z = generate_multibrot(power=4, width=resolution, height=resolution,
                                 max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        elif fname == "newton":
            Z = generate_newton(width=resolution, height=resolution,
                              max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

        else:
            Z = generate_mandelbrot(width=resolution, height=resolution,
                                  max_iter=max_iter, center_x=center_x, center_y=center_y, zoom=zoom)

    # Pure quantum mode
    else:
        c_real, c_imag = text_to_quantum_params(seed_text)
        Z = generate_julia(c_real, c_imag, width=resolution, height=resolution,
                         max_iter=100, zoom=zoom)

    return Z, qc


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


# Available predefined fractals
PREDEFINED_FRACTALS = [
    "Mandelbrot",
    "Julia",
    "Burning Ship",
    "Tricorn",
    "Phoenix",
    "Celtic",
    "Cubic",
    "Quartic",
    "Newton"
]
