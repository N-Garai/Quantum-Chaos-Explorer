# 🌌 Quantum Chaos Explorer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Qiskit](https://img.shields.io/badge/qiskit-2.0+-purple.svg)
![Live Demo](https://quantum-chaos-explorer.streamlit.app/)

An interactive web application that combines quantum computing with fractal mathematics to create stunning visualizations. Explore the intersection of quantum mechanics and chaos theory through 15+ different fractal types, each modulated by quantum circuit measurements.

## ✨ Features

- **15+ Fractal Types**: Mandelbrot, Julia, Burning Ship, Tricorn, Phoenix, Celtic, Cubic, Quartic, Newton, and more
- **Quantum Modulation**: Each fractal is influenced by quantum circuit measurements from Qiskit
- **Text-to-Quantum Seed**: Convert any text into quantum parameters for unique fractal generation
- **Interactive Parameters**: Adjust zoom, iterations, resolution, and quantum influence in real-time
- **High-Quality Export**: Download fractals as PNG images
- **Responsive UI**: Modern Streamlit interface with dark mode support
- **Color Customization**: Multiple color schemes and gradient options

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/N-Garai/Quantum-Chaos-Explorer.git
   cd Quantum-Chaos-Explorer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv qenv
   ```

3. **Activate the virtual environment**
   - Windows (PowerShell):
     ```powershell
     .\qenv\Scripts\Activate.ps1
     ```
   - Windows (CMD):
     ```cmd
     .\qenv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```bash
     source qenv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage

### Basic Workflow

1. **Select a Fractal Type** from the sidebar dropdown
2. **Adjust Parameters**:
   - Resolution: Image quality (128-2048 pixels)
   - Max Iterations: Detail level (50-2000)
   - Zoom & Center: Navigate the fractal space
   - Quantum Influence: Control quantum modulation strength (0-100%)
3. **Enter Quantum Seed** (optional): Any text to generate unique quantum parameters
4. **Generate Fractal**: Click the button to render
5. **Download**: Save your creation as a PNG file

### Advanced Features

- **Julia Set Parameters**: For Julia fractals, customize the complex constant C
- **Color Schemes**: Choose from multiple color palettes
- **Circuit Information**: View the quantum circuit details used in generation
- **Predefined Presets**: Quick access to popular fractal configurations

## 🔬 How It Works

### Quantum-Fractal Integration

1. **Text Seeding**: User input text is hashed and converted into quantum rotation angles
2. **Quantum Circuit**: A 2-qubit circuit is constructed with gates based on the seed
3. **Measurement**: Circuit state is measured to extract complex parameters
4. **Fractal Modulation**: Quantum measurements influence fractal iteration behavior
5. **Rendering**: The fractal is computed point-by-point and colored based on escape time

### Fractal Types

- **Mandelbrot Set**: Classic fractal defined by $z_{n+1} = z_n^2 + c$
- **Julia Set**: Similar to Mandelbrot but with fixed c parameter
- **Burning Ship**: Uses absolute value in iteration: $z_{n+1} = (|Re(z)| + i|Im(z)|)^2 + c$
- **Tricorn**: Anti-holomorphic variant using complex conjugate
- **Phoenix**: Multi-parameter fractal with memory term
- **Celtic**: Mandelbrot with absolute value of real component
- **Newton**: Root-finding based on Newton's method for $z^3 - 1 = 0$
- **And more**: Cubic, Quartic, and other exotic variations

## 📂 Project Structure

```
Quantum-Chaos-Explorer/
├── app.py                      # Main Streamlit application
├── quantum_engine.py           # Fractal generation & quantum logic
├── requirements.txt            # Python dependencies
├── Quantum-Fractal-Demo.ipynb # Jupyter notebook demo
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[Qiskit](https://qiskit.org/)**: Quantum computing framework by IBM
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Matplotlib](https://matplotlib.org/)**: Visualization and plotting

## 📊 Examples

### Quantum-Modulated Mandelbrot
```python
# Generate with seed text
quantum_seed = "chaos theory"
fractal_type = "Mandelbrot"
# Quantum influence: 50%
```

### Custom Julia Set
```python
# Julia with quantum parameters
c_real = -0.4
c_imag = 0.6
fractal_type = "Julia"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**N-Garai**
- GitHub: [@N-Garai](https://github.com/N-Garai)

## 🙏 Acknowledgments

- IBM Qiskit team for the quantum computing framework
- The Streamlit community for the amazing web framework
- Mathematical community for fractal research and algorithms

## 📚 References

- [The Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Fractal Mathematics](https://mathworld.wolfram.com/Fractal.html)
- [Quantum Computing Basics](https://en.wikipedia.org/wiki/Quantum_computing)


## 🔮 Future Enhancements

- [ ] Animation support for fractal zooms
- [ ] 3D fractal rendering
- [ ] More quantum circuit customization options
- [ ] Batch generation and export
- [ ] GPU acceleration for faster rendering
- [ ] Support for custom fractal formulas

---

**Note**: This project is for educational and artistic purposes, demonstrating the creative intersection of quantum computing and fractal mathematics.
