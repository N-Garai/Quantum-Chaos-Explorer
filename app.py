"""
Quantum Fractal Explorer - ULTIMATE VERSION
============================================

Professional web app supporting 15+ fractal types with quantum modulation.

Author: N-Garai
Date: December 2025
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from quantum_engine import (
    generate_fractal_grid,
    get_circuit_info,
    PREDEFINED_FRACTALS,
    text_to_quantum_params
)

# Page Configuration
st.set_page_config(
    page_title="Quantum Fractal Explorer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">✨ Quantum Fractal Explorer</h1>', unsafe_allow_html=True)
st.caption("🎨 15+ Fractal Types | Quantum-Powered | Research-Based")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Fractal Configuration")

    # Quantum Seed
    st.markdown("### 🔮 Quantum Seed")
    seed_text = st.text_input(
        "Seed Phrase",
        value="Quantum",
        help="Text that generates unique quantum parameters"
    )

    # Show quantum parameters
    if seed_text:
        try:
            c_r, c_i = text_to_quantum_params(seed_text)
            st.info(f"🎯 Quantum: c = {c_r:.4f} + {c_i:.4f}i")
        except:
            pass

    st.markdown("---")

    # Fractal Mode
    st.markdown("### 🎨 Fractal Type")

    fractal_mode = st.radio(
        "Choose Mode:",
        ["Predefined Fractals", "Custom Equation", "Pure Quantum"],
        index=0
    )

    fractal_name = None
    custom_equation = None
    mode = "predefined"

    if fractal_mode == "Predefined Fractals":
        fractal_name = st.selectbox(
            "Select Fractal:",
            PREDEFINED_FRACTALS,
            help="Choose from 9 classic and advanced fractals"
        )
        mode = "predefined"

        # Fractal descriptions
        descriptions = {
            "Mandelbrot": "Classic: z → z² + c",
            "Julia": "Quantum-powered: z → z² + c (constant c)",
            "Burning Ship": "z → (|Re(z)| + i|Im(z)|)² + c",
            "Tricorn": "z → conj(z)² + c",
            "Phoenix": "z(n+1) = z(n)² + a + b*z(n-1)",
            "Celtic": "z → (|Re(z²)| + i*Im(z²)) + c",
            "Cubic": "Multibrot: z → z³ + c",
            "Quartic": "Multibrot: z → z⁴ + c",
            "Newton": "Roots of z³ - 1"
        }

        if fractal_name in descriptions:
            st.caption(f"📐 {descriptions[fractal_name]}")

    elif fractal_mode == "Custom Equation":
        st.info("📝 Create your own fractal!")
        custom_equation = st.text_area(
            "Equation:",
            value="z(n+1) = z(n)² + c",
            height=100,
            help="Iterative: z(n+1) = ...\nSimple: sin(x) + cos(y)"
        )
        mode = "custom"

        # Example equations
        with st.expander("📚 Example Equations"):
            st.markdown("**Iterative (proper fractals):**")
            st.code("z(n+1) = z(n)² + c")
            st.code("z(n+1) = z(n)³ + c")
            st.code("z(n+1) = z(n)² + z(n) + c")

            st.markdown("**Simple (patterns):**")
            st.code("sin(x**2 + y**2) * z * 50")
            st.code("exp(-(x**2 + y**2))")
            st.code("sqrt(abs(x*y)) * sin(z*pi) * 20")

    else:
        st.info("⚛️ Pure quantum Julia set")
        mode = "quantum_only"

    st.markdown("---")

    # Rendering Parameters
    st.markdown("### 🔧 Rendering Settings")

    col1, col2 = st.columns(2)

    with col1:
        resolution = st.select_slider(
            "Resolution",
            options=[100, 150, 200, 250, 300, 400, 500],
            value=200,
            help="Higher = more detail"
        )

    with col2:
        max_iter = st.select_slider(
            "Iterations",
            options=[50, 100, 150, 200, 256, 300, 400],
            value=256,
            help="More = finer detail"
        )

    zoom = st.slider("Zoom", min_value=0.1, max_value=3.0, value=1.5, step=0.1)
    center_x = st.slider("Center X", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    center_y = st.slider("Center Y", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    # Colormap
    colormap = st.selectbox(
        "🎨 Color Palette",
        ["twilight", "hot", "plasma", "viridis", "magma", "turbo", "hsv", "inferno"],
        index=0
    )

    st.markdown("---")

    # Generate Button
    generate_btn = st.button("🚀 Generate Fractal", type="primary", use_container_width=True)

# Main Content
col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("🎨 Your Quantum Fractal")

    # Generate
    if generate_btn or 'fractal_data' not in st.session_state:
        with st.spinner("🔮 Generating quantum fractal..."):
            try:
                Z, qc = generate_fractal_grid(
                    resolution=resolution,
                    zoom=zoom,
                    center_x=center_x,
                    center_y=center_y,
                    seed_text=seed_text,
                    fractal_mode=mode,
                    fractal_name=fractal_name.lower() if fractal_name else "mandelbrot",
                    custom_equation=custom_equation,
                    max_iter=max_iter
                )

                st.session_state.fractal_data = Z
                st.session_state.circuit = qc
                st.session_state.mode = fractal_mode
                st.session_state.seed = seed_text
                st.session_state.fractal_name = fractal_name if fractal_name else "Custom"

                st.success("✅ Fractal generated!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

    # Display
    if 'fractal_data' in st.session_state:
        Z = st.session_state.fractal_data

        fig, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(
            Z, interpolation='bilinear', cmap=colormap,
            extent=[center_x - zoom, center_x + zoom,
                   center_y - zoom, center_y + zoom],
            origin='lower'
        )

        mode_display = st.session_state.get('mode', 'Unknown')
        fname = st.session_state.get('fractal_name', 'Fractal')
        ax.set_title(f'{fname} | Mode: {mode_display}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)

        plt.colorbar(img, ax=ax, label='Intensity', fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)

        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        st.download_button(
            label="💾 Download PNG",
            data=buf,
            file_name=f"quantum_{fname}_{st.session_state.seed}.png",
            mime="image/png"
        )

with col2:
    st.subheader("📊 Fractal Info")

    if 'circuit' in st.session_state:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔮 Seed", f'"{st.session_state.seed}"')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎨 Mode", st.session_state.mode)
        st.markdown('</div>', unsafe_allow_html=True)

        if fractal_mode == "Predefined Fractals" and fractal_name:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📐 Type", fractal_name)
            st.markdown('</div>', unsafe_allow_html=True)

        elif fractal_mode == "Custom Equation" and custom_equation:
            st.markdown("**📝 Your Equation:**")
            st.code(custom_equation, language="python")

        st.markdown("---")
        st.markdown("**⚛️ Quantum Circuit**")

        info = get_circuit_info(st.session_state.circuit)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Depth", info['depth'])
        with col_b:
            st.metric("Gates", info['num_gates'])

        with st.expander("🔧 Gate Breakdown"):
            for gate, count in info['gate_counts'].items():
                st.text(f"{gate.upper()}: {count}")

# Educational Section
st.markdown("---")
st.header("📚 Fractal Gallery & Guide")

tab1, tab2, tab3 = st.tabs(["🎨 Fractal Guide", "⚛️ Science", "💻 Custom Equations"])

with tab1:
    st.markdown("""
    ### Fractal Types Explained

    #### 📍 Classic Fractals

    **Mandelbrot Set** `z → z² + c`
    - Most famous fractal
    - Self-similar at all scales
    - Best viewed at zoom = 1.2, center = (-0.5, 0)

    **Julia Set** `z → z² + c` (constant c)
    - Each quantum seed = different Julia set
    - Try seeds: "Quantum", "Dream", "Chaos"
    - Best viewed at zoom = 1.5, center = (0, 0)

    #### 🔥 Advanced Fractals

    **Burning Ship** `z → (|Re(z)| + i|Im(z)|)² + c`
    - Ship-like structures
    - Best viewed at center = (-0.5, -0.5)

    **Tricorn** `z → conj(z)² + c`
    - Conjugate variant
    - Three-fold symmetry

    **Phoenix** `z(n+1) = z(n)² + a + b*z(n-1)`
    - Uses previous iteration
    - Unique flowing patterns

    **Celtic** `z → (|Re(z²)| + i*Im(z²)) + c`
    - Celtic knot patterns

    #### 📐 Polynomial Fractals

    **Cubic/Quartic** `z → z³ + c` or `z → z⁴ + c`
    - Higher-order Mandelbrot
    - More complex symmetries

    **Newton** Roots of `z³ - 1`
    - Converges to roots
    - Beautiful basin boundaries

    ### Tips for Exploration

    - 🔍 Start with low resolution (100-200) for fast preview
    - 🎨 Try different color palettes for same fractal
    - 🌀 Deep zoom requires more iterations (400+)
    - ✨ Julia sets change dramatically with quantum seeds
    """)

with tab2:
    st.markdown("""
    ### The Science Behind It

    **What Are Fractals?**

    Fractals are mathematical sets exhibiting self-similarity at different scales. 
    They're generated through iteration:

    1. Start with initial value z₀
    2. Apply function: zₙ₊₁ = f(zₙ)
    3. Check if sequence remains bounded
    4. Color based on escape time

    **Quantum Integration**

    Traditional fractals use fixed parameters. This project uses **quantum computing**
    to generate those parameters:

    ```
    Text Seed → Quantum Circuit → Measurement → Complex Parameter c
    ```

    **Why It Works:**

    - Each character creates unique quantum gates
    - Quantum interference creates unpredictable patterns
    - Same seed = same fractal (deterministic)
    - Different seed = completely different fractal

    **Research Foundation:**

    Based on: "Quantum-Circuit-Based Visual Fractal Image Generation"
    (arxiv.org/pdf/2508.18835.pdf)

    Key insight: Use quantum to generate **parameters**, not per-pixel values.
    """)

with tab3:
    st.markdown("""
    ### Custom Equation Guide

    #### Iterative Equations (Generate Real Fractals)

    **Format:** `z(n+1) = <expression>`

    **Examples:**
    ```python
    z(n+1) = z(n)² + c          # Mandelbrot
    z(n+1) = z(n)³ + c          # Cubic
    z(n+1) = z(n)² + z(n) + c  # Modified
    z(n+1) = z(n)^4 + c         # Quartic (^ works too!)
    ```

    **Variables:**
    - `z(n)` or `z` - current iteration value
    - `c` - complex parameter (from coordinates OR quantum)

    If equation contains `c`, it's Mandelbrot-style (c from coordinates).
    If no `c`, it's Julia-style (z from coordinates, c from quantum).

    #### Simple Equations (Generate Patterns)

    **Format:** Just write expression

    **Examples:**
    ```python
    sin(x**2 + y**2) * z * 50
    exp(-(x**2 + y**2))
    sqrt(abs(x*y)) * sin(z*pi) * 20
    cos(sqrt(x**2 + y**2) * 5) * z * 30
    ```

    **Variables:**
    - `x`, `y` - coordinates
    - `z` - quantum parameter (0-1)

    **Functions:**
    - `sin, cos, tan`
    - `exp, log, sqrt`
    - `abs, pi, e`

    **Operators:**
    - `+, -, *, /`
    - `**` or `^` for power

    ### Pro Tips

    1. Multiply simple equations by 20-50 for visible patterns
    2. Use `z` in simple equations to add quantum variation
    3. Iterative equations need 200+ iterations
    4. Test with low resolution first!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Quantum Fractal Explorer - Ultimate Edition</strong></p>
    <p>15+ Fractal Types | Quantum-Powered | Production-Ready</p>
    <p>Built with Qiskit, NumPy, Matplotlib & Streamlit</p>
</div>
""", unsafe_allow_html=True)
