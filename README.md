# Five‑Qubit [[5,1,3]] Code Implementation with Marimo + Qiskit

A Marimo notebook demonstrating implementation of the 5‑qubit perfect quantum error‑correcting code ([[5,1,3]] code). This notebook:

1. Prepares logical states $\ket{0_L}$ and $\ket{1_L}$  
2. Introduces single‑qubit Pauli errors ($X$, $Y$, $Z$)  
3. Performs syndrome measurement using four stabilizers  
4. Visualizes the quantum circuit  
5. Collects and displays syndrome results in a DataFrame  
6. Applies error correction via lookup table  
7. Measures the logical qubit and shows results

Check out the html to look at the Marimo notebook or the png image to see the circuit introducing a $Y$ (bit-flip) error at $q_2$

![5-qubit_code](https://github.com/user-attachments/assets/15178ee7-cd1d-4f30-bc83-a68e02834f13)
 
## Contents

- **Preparation of logical states**  
  Initializes the 5‑qubit register into logical $\ket{0_L}$ or $\ket{1_L}$.

- **Error injection**  
  Applies every Pauli error on each physical qubit to generate unique syndromes.

- **Syndrome measurement**  
  Implements function `measure_syndrome(circ, data_qubits, ancilla_qubits)` building four stabilizer measurements and returns a 4‑bit syndrome.

- **Lookup‑table and correction**  
  Builds a mapping from syndrome → correction (Pauli, qubit index). Applies this in the “error‑corrected experiment”.

- **Logical measurement**  
  Measures $\braket{Z_L}$ = $Z^{\otimes5}$ on corrected state to verify correction.

## Usage

1. Clone repo  
 ```bash
   git clone https://github.com/secpriano/5-qubit_code.git
   cd 5-qubit_code
```
2. Install dependencies
```bash
pip install uv
uv venv
```
3. Activate the virtual environment depending on your operating system:

On Linux / macOS:
```bash
source .venv/bin/activate
```
On Windows (CMD):
```bash
.venv\Scripts\activate.bat
```
On Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```
4. Install marimo (notebook) 
```bash
uv add marimo
```
3. Open and edit notebook
```bash
marimo edit
```
4. Install dependencies in notebook and run it

## Explanation
For deeper understanding, see sections:

- Running the Syndrome Experiment – step‑by‑step walkthrough
- Running Error‑Corrected Experiment – logic of qc.switch, logical Z measurement, outcome interpretation

References

- GitHub: bernwo/five‑qubit‑code
- MIT 8.371 Lecture 6
- Preskill Lecture 7
- Laflamme et al., “Arbitrary accurate quantum computation via the five‑qubit code” (arXiv:1010.3242)

Let me know if I made a mistake.
