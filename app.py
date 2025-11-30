"""
Quantum-Chaos-Explorer: Enhanced Interactive Web Application
============================================================
Now supports USER-DEFINED FRACTAL EQUATIONS and predefined fractals!

Author: N-Garai
Date: November 2025
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from quantum_engine import (
    generate_fractal_grid, 
    get_circuit_info,
    PREDEFINED_FRACTALS
)

# Professional Page Configuration
st.set_page_config(
    page_title="Quantum-Chaos-Explorer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Quantum-Chaos-Explorer ✨</p>', unsafe_allow_html=True)
st.caption("🎨 Create quantum-classical hybrid fractals with custom equations or choose from classics!")

# Sidebar: User Controls
with st.sidebar:
    st.header("⚙️ Fractal Configuration")

    # Quantum Seed
    seed_text = st.text_input(
        "🔮 Quantum Seed Phrase", 
        value="Quantum",
        help="This text generates a unique quantum circuit that modulates your fractal"
    )

    st.markdown("---")
    st.subheader("🎨 Fractal Type Selection")

    # Fractal Mode Selection
    fractal_mode = st.radio(
        "Choose Fractal Mode:",
        ["Predefined Fractals", "Custom Equation", "Pure Quantum"],
        index=0
    )

    fractal_name = None
    custom_equation = None

    if fractal_mode == "Predefined Fractals":
        fractal_name = st.selectbox(
            "Select a Classic Fractal:",
            PREDEFINED_FRACTALS,
            help="Choose from famous mathematical fractals"
        )
        mode = "predefined"

    elif fractal_mode == "Custom Equation":
        st.info("📝 Enter your own fractal equation!")
        custom_equation = st.text_area(
            "Equation (use x, y, z):",
            value="sin(x**2 + y**2) * z",
            help="Use variables: x, y (coordinates), z (quantum probability)\n"
                 "Functions: sin, cos, tan, exp, log, sqrt, abs, pi, e\n"
                 "Example: exp(-(x**2 + y**2)) * cos(z*10)"
        )
        mode = "custom"

        # Show example equations
        with st.expander("📚 Example Equations"):
            st.code("sin(x) + cos(y) * z")
            st.code("exp(-(x**2 + y**2))")
            st.code("x**2 - y**2 + z")
            st.code("sqrt(abs(x*y)) * sin(z*pi)")
            st.code("cos(sqrt(x**2 + y**2) * 5) * z")

    else:  # Pure Quantum
        st.info("⚛️ Pure quantum interference mode")
        mode = "quantum_only"

    st.markdown("---")
    st.subheader("🔧 Rendering Parameters")

    # Resolution and zoom controls
    resolution = st.slider(
        "Resolution (pixels)", 
        min_value=50, 
        max_value=300, 
        value=120, 
        step=10,
        help="Higher resolution = more detail but slower rendering"
    )

    zoom = st.slider(
        "Zoom Level", 
        min_value=0.1, 
        max_value=3.0, 
        value=1.5, 
        step=0.1
    )

    center_x = st.slider(
        "Center X", 
        min_value=-2.0, 
        max_value=2.0, 
        value=0.0, 
        step=0.1
    )

    center_y = st.slider(
        "Center Y", 
        min_value=-2.0, 
        max_value=2.0, 
        value=0.0, 
        step=0.1
    )

    # Colormap selection
    colormap = st.selectbox(
        "Color Palette",
        ["twilight", "magma", "viridis", "plasma", "inferno", "turbo", "rainbow"],
        index=0
    )

    st.markdown("---")
    generate_btn = st.button("🚀 Generate Fractal", type="primary", use_container_width=True)

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 Your Quantum Fractal")

    if generate_btn or 'fractal_data' not in st.session_state:
        with st.spinner("🔮 Computing quantum-fractal hybrid..."):
            try:
                Z, qc = generate_fractal_grid(
                    resolution=resolution,
                    zoom=zoom,
                    center_x=center_x,
                    center_y=center_y,
                    seed_text=seed_text,
                    fractal_mode=mode,
                    fractal_name=fractal_name.lower() if fractal_name else None,
                    custom_equation=custom_equation
                )

                st.session_state.fractal_data = Z
                st.session_state.circuit = qc
                st.session_state.mode = fractal_mode

            except Exception as e:
                st.error(f"❌ Error generating fractal: {str(e)}")
                st.stop()

    # Display fractal
    if 'fractal_data' in st.session_state:
        Z = st.session_state.fractal_data

        fig, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(
            Z,
            interpolation='bilinear',
            cmap=colormap,
            extent=[center_x - zoom, center_x + zoom, 
                   center_y - zoom, center_y + zoom],
            origin='lower'
        )

        mode_display = st.session_state.get('mode', 'Unknown')
        ax.set_title(f"Fractal Mode: {mode_display}", fontsize=16, pad=20)
        ax.set_xlabel("X coordinate", fontsize=12)
        ax.set_ylabel("Y coordinate", fontsize=12)

        # Add colorbar
        plt.colorbar(img, ax=ax, label="Intensity", fraction=0.046, pad=0.04)

        st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("📊 Fractal Metadata")

    if 'circuit' in st.session_state:
        info = get_circuit_info(st.session_state.circuit)

        st.metric("Quantum Seed", f'"{seed_text}"')
        st.metric("Fractal Mode", st.session_state.get('mode', 'N/A'))

        if fractal_mode == "Predefined Fractals":
            st.metric("Fractal Type", fractal_name)
        elif fractal_mode == "Custom Equation":
            st.code(custom_equation, language="python")

        st.markdown("---")
        st.markdown("**⚛️ Quantum Circuit Info**")
        st.metric("Circuit Depth", info['depth'])
        st.metric("Total Gates", info['num_gates'])

        with st.expander("Gate Breakdown"):
            for gate, count in info['gate_counts'].items():
                st.text(f"{gate.upper()}: {count}")

# Educational Section
st.markdown("---")
st.header("📚 Understanding Quantum Fractals")

tab1, tab2, tab3 = st.tabs(["🎨 What Are These?", "⚛️ The Physics", "💡 Innovation"])

with tab1:
    st.markdown("""
    ### What You're Seeing

    These images are **quantum-classical hybrid fractals** — a unique combination of:

    1. **Classical Fractal Mathematics**: Traditional fractals like Mandelbrot, Julia sets, or your custom equation
    2. **Quantum Modulation**: Each pixel's color is influenced by a quantum circuit generated from your seed text

    The result? Fractals that have **quantum fingerprints** — subtle interference patterns and probability-based 
    variations that make each seed phrase produce a mathematically unique artwork.

    **Try This:**
    - Change just one letter in your seed phrase and watch how the pattern transforms
    - Switch between fractals to see how quantum mechanics interacts differently with each mathematical structure
    """)

with tab2:
    st.markdown("""
    ### The Physics Behind It

    **Quantum Circuit Generation:**
    - Your text input is converted into a sequence of quantum gates (Hadamard, Rotation, Phase gates)
    - Each character's ASCII value determines which gate is applied
    - This creates a unique quantum state for every possible text input

    **Fractal Computation:**
    - For each pixel at coordinates (x, y), we:
      1. Map x and y to quantum rotation angles (θ and φ)
      2. Apply those rotations to our text-generated circuit
      3. Measure the quantum probability distribution
      4. Use that probability to modulate the classical fractal equation

    **The Magic:**
    - Quantum interference causes the swirls and unexpected patterns
    - The classical fractal provides the overall structure
    - The combination creates art that couldn't exist without quantum mechanics!

    \[ P(x,y) = |\langle 0 | U(\theta, \phi) | \psi_{\text{seed}} \rangle|^2 \times f_{\text{fractal}}(x,y) \]
    """)

with tab3:
    st.markdown("""
    ### Innovation & Interview Talking Points

    **What Makes This Project Unique:**

    1. **Text-to-Circuit Compiler**: Deterministic mapping from natural language to quantum circuits
    2. **Hybrid Algorithm**: First-of-its-kind fusion of quantum computing and classical fractal theory
    3. **Interactive Experimentation**: Real-time quantum simulation with visual feedback
    4. **Custom Equation Parser**: Safe mathematical expression evaluator for user-defined fractals

    **Technical Achievements:**
    - Efficient Statevector simulation (avoiding expensive circuit execution)
    - Modular architecture separating quantum logic from visualization
    - Professional-grade error handling and input validation

    **In an Interview, Say:**
    > "I built a quantum-classical hybrid fractal generator that demonstrates quantum chaos theory. 
    > Users can input custom mathematical equations that get modulated by quantum interference patterns. 
    > The quantum circuits are generated from text input, making every seed phrase produce a unique 
    > quantum fingerprint. This project showcases both my quantum computing expertise and software 
    > engineering skills in building production-ready applications."
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Quantum-Chaos-Explorer</strong> | Built with Qiskit, Streamlit & NumPy</p>
    <p>A portfolio project by a Senior Quantum Engineer</p>
</div>
""", unsafe_allow_html=True)
