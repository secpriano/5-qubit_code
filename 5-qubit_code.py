import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Five-Qubit [[5,1,3]] Code Implementation

    This Marimo notebook demonstrates a implementation of the 5-qubit perfect code (also known as the [[5,1,3]] code). It covers:

    1. Preparation of logical states $\ket{0_L}$ and $\ket{1_L}$
    2. Introduction of single-qubit Pauli errors
    3. Syndrome measurement via four stabilizers
    4. Visualization of the circuit
    5. Collection and display of syndrome results in a DataFrame
    6. Apply correction based on the lookup table from the DataFrame
    7. Measure the logical qubit
    8. Show results

    ### Go to "Running the Syndrome Experiment" for explanation
    ### Go to "Running Error-Corrected Experiment" for explanation

    References:

    - https://github.com/bernwo/five-qubit-code
    - MIT Lecture Notes: 8.371, Lecture 6
    - Preskill’s Lecture 7
    - Laflamme et al., arXiv:1010.3242
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Imports and Backend Setup""")
    return


@app.cell
def _():
    import pandas as pd
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import Initialize
    from IPython.display import display
    return (
        AerSimulator,
        ClassicalRegister,
        Initialize,
        QiskitRuntimeService,
        QuantumCircuit,
        QuantumRegister,
        display,
        pd,
        transpile,
    )


@app.cell
def _(AerSimulator, QiskitRuntimeService):
    service = QiskitRuntimeService(channel='local')

    # Specify a QPU to use for the noise model
    simulator = AerSimulator()
    return (simulator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Syndrome Measurement

    Define a function to measure the four stabilizers S₀…S₃ using 4 ancilla qubits. After preparing the ancillas in |+> states, controlled gates entangle ancillas with data qubits according to each stabilizer pattern. Finally, ancillas are returned to the Z-basis and measured.

    ```python
    measure_syndrome(circ, data_qubits, ancilla_qubits)
    ```
    """
    )
    return


@app.cell
def _(QuantumCircuit):
    def measure_syndrome(circ: QuantumCircuit, physical_qubits, ancilla_qubits):
        circ.h(ancilla_qubits)
        circ.barrier()

        # S0 = Z X X Z I on [0,1,2,3]
        circ.cz(ancilla_qubits[3], physical_qubits[0])
        circ.cx(ancilla_qubits[3], physical_qubits[1])
        circ.cx(ancilla_qubits[3], physical_qubits[2])
        circ.cz(ancilla_qubits[3], physical_qubits[3])
        circ.barrier()

        # S1 = X X Z I Z on [0,1,2,4]
        circ.cx(ancilla_qubits[2], physical_qubits[0])
        circ.cx(ancilla_qubits[2], physical_qubits[1])
        circ.cz(ancilla_qubits[2], physical_qubits[2])
        circ.cz(ancilla_qubits[2], physical_qubits[4])
        circ.barrier()

        # S2 = X Z I Z X on [0,1,3,4]
        circ.cx(ancilla_qubits[1], physical_qubits[0])
        circ.cz(ancilla_qubits[1], physical_qubits[1])
        circ.cz(ancilla_qubits[1], physical_qubits[3])
        circ.cx(ancilla_qubits[1], physical_qubits[4])
        circ.barrier()

        # S3 = Z I Z X X on [0,2,3,4]
        circ.cz(ancilla_qubits[0], physical_qubits[0])
        circ.cz(ancilla_qubits[0], physical_qubits[2])
        circ.cx(ancilla_qubits[0], physical_qubits[3])
        circ.cx(ancilla_qubits[0], physical_qubits[4])
        circ.barrier()

        # Return and measure
        circ.h(ancilla_qubits)
        circ.barrier()

        circ.measure(ancilla_qubits, range(len(ancilla_qubits)))
    return (measure_syndrome,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Logical State Initialization

    The following helper function appends an instruction to initialize a 5-qubit register into one of the two logical basis states |0>ₗ or |1>ₗ.

    ```python
    init_logical_state(circ, qubits, state)
    ```

    where `state` is either `'0'` or `'1'`.
    """
    )
    return


@app.cell
def _(display):
    from qiskit.quantum_info import Statevector
    import numpy as np

    vectors = {
        '0': [
            1/4, 0, 0, 1/4, 0, -1/4, 1/4, 0,
            0, -1/4, -1/4, 0, 1/4, 0, 0, -1/4,
            0, 1/4, -1/4, 0, -1/4, 0, 0, -1/4,
            1/4, 0, 0, -1/4, 0, -1/4, -1/4, 0
        ],
        '1': [
            0, -1/4, -1/4, 0, -1/4, 0, 0, 1/4,
            -1/4, 0, 0, -1/4, 0, -1/4, 1/4, 0,
            -1/4, 0, 0, 1/4, 0, -1/4, -1/4, 0,
            0, 1/4, -1/4, 0, 1/4, 0, 0, 1/4
        ]
    }


    zero_L = Statevector(np.array(vectors['0'], dtype=complex))
    one_L  = Statevector(np.array(vectors['1'], dtype=complex))

    display(zero_L.draw("latex"))
    display(one_L.draw("latex"))
    return (vectors,)


@app.cell
def _(Initialize, QuantumCircuit, vectors):
    def init_logical_state(circ: QuantumCircuit, qubits, state: str):
        inst = Initialize(vectors[state])
        inst.label = f"|{state}_L>"
        circ.append(inst, qubits)
    return (init_logical_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Running the Syndrome Experiment

    Loop over each Pauli error (X, Z, Y) on each data qubit, append the error, measure the syndrome, and collect results in a `pandas.DataFrame`.

    ### Step-by-step in code comments
    """
    )
    return


@app.cell
def _(
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    display,
    init_logical_state,
    measure_syndrome,
    pd,
    simulator,
    transpile,
):
    def run_syndrome_experiment():
        labels = [f"X[{i}]" for i in range(5)] + [f"Z[{i}]" for i in range(5)] + [f"Y[{i}]" for i in range(5)]
        syndromes = []

        for label in labels:
            # Prepare fresh circuit
            # this register has 5 qubits which will be the logical qubit
            physical_qubits = QuantumRegister(5, 'q')
            # this register has 4 qubits which will be the ancillas for the syndrome measurement
            ancilla_qubits = QuantumRegister(4, 'a')
            # this register has 4 (calssical!) bits which is where the syndrome measurement will be saved
            syndrome_bits = ClassicalRegister(4, 's')
            # Add these register together
            qc = QuantumCircuit(physical_qubits, ancilla_qubits, syndrome_bits)
        
            # Initialize a |0_L>, Logical qubit 0 in 5 physical qubits
            init_logical_state(qc, physical_qubits, '0')
            qc.barrier()

            # See 1st figure that init a |0_L>
            if label == "X[0]":
                display("Initialize a 5-qubit code circuit with a |0_L> state")
                display(qc.draw('mpl'))
                  
            # Inject X, Y, and Z error at each physical qubit location
            pauli, tgt = label[0], int(label[2])
            getattr(qc, pauli.lower())(physical_qubits[tgt])
            qc.barrier()

            # See 2nd figure that introduces an X error at q_0
            if label == "X[0]":
                display("introduces an X error at q_0")
                display(qc.draw('mpl'))

            # Measure syndrome
            measure_syndrome(qc, physical_qubits, ancilla_qubits)

            # See 3rd figure that adds the 4 stabilizers and measure it
            if label == "X[0]":
                display("adds the 4 stabilizers and measure it")
                display(qc.draw('mpl'))

            # Transpile & run
            result = simulator.run(transpile(qc, simulator)).result()
            bit = list(result.get_counts().keys())[0]
            syndromes.append(bit)

            # Draw circuit
            display(f"Complete circuit for {label} error")
            display(qc.draw('mpl'))

        df = pd.DataFrame(syndromes, index=labels, columns=['Syndrome'])
        return df
    return (run_syndrome_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Execute and display results""")
    return


@app.cell
def _(run_syndrome_experiment):
    df = run_syndrome_experiment()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Syndrome measurement result
    Errors on the physical qubits can be detected via stabilizer measurements.

    After each type of error at each location has been measured (3 Pauli gates $\times$ 5 physical qubit $=$ 15 errors), we will get the table below.

    Notice that each error and there location has a unique bitstring.

    We can create a look up table from these unique bitstring to correct a code.

    If we place an $X$ error at $q_0$, we can correct it by placing the same Pauli type, so another $X$. same goes for $Y$, and $Z$.

    But normally when you run a code you don't know where an error occurs, so you look at what the bistring is to match with the look up table to see which gate to apply and where.

    For example syndrome measurement shows $1110$, If we look at the table below we see the error is Pauli $Y$ at $q_1$. So we apply the $Y$ gate at $q_1$ to correct it.
    """
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### How did we create this bitstring?

    The 4 stabilizers are:

    - $ZXXZI$
    - $XXZIZ$
    - $XZIZX$
    - $ZIZXX$

    We introduce an $X$ error at $q_0$:

    - $XIIII$

    We get each bit from the bitstring (syndrome) by calculating what errors commute with the stabilizers.

    Meaning that $[A, B] = AB − BA = 0$. If it's not $0$, $1$

    or

    For each qubit position:

    - If one operator is I → skip.
    - If both are the same Pauli (X vs X, Z vs Z) → that part commutes.
    - If they are X vs Z or Z vs X → that part anticommutes (counts 1).

    Sum anticommutes:

    - Even → overall commute → syndrome 0.
    - Odd → overall anticommute → syndrome 1.

    $[XIIII, ZXXZI] = 1$

    $[XIIII, XXZIZ] = 0$

    $[XIIII, XZIZX] = 0$

    $[XIIII, ZIZXX] = 1$

    We can rewrite above by just looking at the column where the $X$ error is.

    $[X, Z] = XZ - ZX = 1$

    $[X, X] = XX - XX = 0$

    $[X, X] = XX - XX = 0$

    $[X, Z] = XZ - ZX = 1$


    Do an example for $Z$ error at $q_2$:

    - $IIZII$

    Answer:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Syndrome Lookup & Correction

    Map each 4-bit syndrome to the corresponding corrective Pauli on one of the 5 data qubits.
    """
    )
    return


@app.cell
def _():
    syndrome_to_error = {
        '1001': ('X', 0), '0010': ('X', 1), '0101': ('X', 2), '1010': ('X', 3), '0100': ('X', 4),
        '0110': ('Z', 0), '1100': ('Z', 1), '1000': ('Z', 2), '0001': ('Z', 3), '0011': ('Z', 4),
        '1111': ('Y', 0), '1110': ('Y', 1), '1101': ('Y', 2), '1011': ('Y', 3), '0111': ('Y', 4),
        '0000': (None, None)  # no error
    }
    return (syndrome_to_error,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Running Error-Corrected Experiment

    Loop over Pauli errors, measure syndrome, apply correction, then measure logical qubit to verify.
    """
    )
    return


@app.cell
def _(syndrome_to_error):
    from qiskit.circuit.controlflow import SwitchCaseOp

    # syndromes (0–15) map to Pauli corrections (pauli, qubit_idx)
    cases = [
        (int(s, 2), [pauli, idx])
        for s, (pauli, idx) in syndrome_to_error.items() if pauli is not None
    ]
    return (cases,)


@app.function
def logical_z(qc, physical_qubits):
    qc.h(0)
    qc.cz(physical_qubits, 0)
    qc.h(0)


@app.cell
def _(
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    cases,
    display,
    init_logical_state,
    measure_syndrome,
    pd,
    simulator,
    transpile,
):
    def run_corrected_experiment():
        # Prepare fresh circuit
        # this register has 1 qubit to measure the logical qubit
        logical_qubit = QuantumRegister(1, 'l');
        # this register has 5 qubits which will be the logical qubit
        physical_qubits = QuantumRegister(5, 'q')
        # this register has 4 qubits which will be the ancillas for the syndrome measurement
        ancilla_qubits = QuantumRegister(4, 'a')
        # this register has 4 (calssical!) bits which is where the syndrome measurement will be saved
        syndrome_bits = ClassicalRegister(4, 's')
        # this register has 1 (calssical!) bit which is where the logical measurement will be saved
        logical_bit = ClassicalRegister(1, 'L')
        qc = QuantumCircuit(logical_qubit, physical_qubits, ancilla_qubits, syndrome_bits, logical_bit)

        # Initialize a |0_L>, Logical qubit 0 in 5 physical qubits
        init_logical_state(qc, physical_qubits, '0')
        qc.barrier()

        # Inject error
        qc.y(3)
        qc.barrier()

        # Measure syndrome
        measure_syndrome(qc, physical_qubits, ancilla_qubits)
        qc.barrier()

        # Switch-based syndrome correction for Pauli type and location
        # From the syndrome measurement, we look up the bitstring and apply the Pauli gate correction
        with qc.switch(syndrome_bits) as case:
            for s_int, (pauli, idx) in cases:
                with case(s_int):
                    getattr(qc, pauli.lower())(physical_qubits[idx])

        qc.barrier()
    
        # Apply logical Z (ZZZZZ)
        logical_z(qc, physical_qubits)

        # Measure logical qubit
        qc.measure(logical_qubit, logical_bit)
        display(qc.draw('mpl', interactive=True))

        # Run
        job = simulator.run(transpile(qc, simulator))
        counts = job.result().get_counts()
        return pd.DataFrame([counts])
    return (run_corrected_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Example execution""")
    return


@app.cell
def _(run_corrected_experiment):
    df_corr = run_corrected_experiment()  
    return (df_corr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A logical qubit has been measured in the computational basis by performing a parity measurement on $\bar {Z} = ZZZZZ$. If the measured ancilla is $0$, the logical qubit is $\ket{0_L}$. If the measured ancilla is $1$, the logical qubit is $\ket{1_L}$.

    The dataframe result shows:

    - column name: Bit $\in \{0 , 1\}$ and a bitstring of the correction string
    - value: Amount shots runned (total 1024)

    Correct result should have 1 column (all shots) with the correct error correction and logical measurement
    """
    )
    return


@app.cell
def _(df_corr):
    df_corr
    return


if __name__ == "__main__":
    app.run()
