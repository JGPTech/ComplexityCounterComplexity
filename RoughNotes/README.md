# Quantum Notes Dump — Complexity / Entanglement / Control

This repo contains three short papers (LaTeX notes) and companion code demos.  
The goal is not polish but to keep the notes + runnable examples in one place.

---

## Part I — Complexity / Counter-Complexity Framework
**Paper:** `notes/complexity_countercomplexity.tex`  
**Code:** `code/qutip/jc_energy_mapping.py`  

- Jaynes–Cummings model used as a demo.  
- Splits energy into **complexity (qubit)**, **counter-complexity (cavity)**, and **interaction**.  
- Produces CSV + PNG (`jc_energy_timeseries.csv`, `jc_energy_mapping.png`).  

Run:
```bash
python code/qutip/jc_energy_mapping.py
````

---

## Part II — Entanglement via Environmental Mediation

**Paper:** `notes/entanglement_env.tex`
**Code:** `code/qutip/entanglement_env_proxy.py`

* Two-qubit entanglement under environment-mediated dephasing.
* Shows concurrence, ⟨Z₁Z₂⟩, and energy balance with environment proxy `A_e(t)`.
* Outputs CSV + PNG (`entanglement_timeseries.csv`, `entanglement_env_proxy.png`).

Run:

```bash
python code/qutip/entanglement_env_proxy.py
```

---

## Part III — Control-Aware Noise / Error-Corrected AND Gate

**Paper:** `notes/control_aware_noise.tex`
**Code:** `code/qiskit/part3_and_demo.py`

* Qiskit/Aer demo with staged progression:

  1. Encode only
  2. Encode + AND
  3. Syndrome (no correction)
  4. Syndrome + correction
  5. Error injection
  6. Application demo (`O = A AND B`)

Run:

```bash
python code/qiskit/part3_and_demo.py
```

---

## Requirements

For QuTiP parts:

```
qutip>=4.7
matplotlib>=3.7
numpy>=1.23
```

For Qiskit part:

```
qiskit==1.2.*
qiskit-aer==0.15.*
numpy>=1.23
```

---

## Notes

* All three scripts are headless-safe (plots saved to PNG, data to CSV).
* The notes and code are aligned: equations in the LaTeX papers have direct counterparts in the scripts.
* Purpose: **archive & share ideas quickly**, not a polished library.

```
