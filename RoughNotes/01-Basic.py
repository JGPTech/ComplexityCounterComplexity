#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jaynes–Cummings Energy Decomposition (QuTiP)
--------------------------------------------
This script is the runnable companion for your Part I note:
“Complexity / Counter-Complexity in Quantum Network Systems.”

Mapping to the note:
- Complexity operator (qubit sector):        O_C   ~ σ_z   (we use H_q = (ω_q/2) σ_z)
- Counter-Complexity operator (cavity field):O_\barC ~ a†a (we use H_c = ω_c a†a)
- Interaction operator:                       I    = a† σ_- + a σ_+

Energy split:
    E_total = ⟨H⟩ = ⟨H_q⟩ + ⟨H_c⟩ + ⟨H_int⟩
which corresponds to your boxed equation
    E_total = E_complexity + E_counter-complexity + E_interaction

We evolve a simple JC system on resonance with one photon in the cavity and
the qubit in |g⟩. Energy sloshes between “complexity” (qubit) and
“counter-complexity” (field) via the interaction term. We verify the energy
balance to numerical precision and save a PNG + CSV for GitHub.

Outputs:
  * PNG:  jc_energy_mapping.png
  * CSV:  jc_energy_timeseries.csv
"""

import numpy as np
from qutip import (
    basis, qeye, tensor, destroy, sigmap, sigmam, sigmaz, mesolve, expect
)

# ----------------------------- Headless plotting -----------------------------
import matplotlib
# Use Agg so this runs on servers/CI; we'll still try to show() if possible.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# 1) PARAMETERS  (ties to “Hamiltonian and Energy Mapping” section)
# =============================================================================
wc   = 1.0     # cavity frequency (ω_\barC)
wq   = 1.0     # qubit frequency  (ω_C)
g    = 0.05    # interaction strength  (coupling)
N    = 6       # cavity Fock cutoff (truncation)
tlist = np.linspace(0, 50, 600)  # time grid (arb. units)

# =============================================================================
# 2) OPERATORS (Complexity, Counter-Complexity, Interaction)
# =============================================================================
a      = destroy(N)         # cavity lowering
I_cav  = qeye(N)
I_qub  = qeye(2)
sm     = sigmam()           # qubit lowering
sp     = sigmap()
sz     = sigmaz()

# Embed into the total Hilbert space: cavity ⊗ qubit
a   = tensor(a, I_qub)
ad  = a.dag()
sm  = tensor(I_cav, sm)
sp  = tensor(I_cav, sp)
sz  = tensor(I_cav, sz)
num = ad * a                               # photon number operator a†a
P_e  = tensor(I_cav, basis(2, 1)*basis(2, 1).dag())  # |e⟩⟨e| projector (for diagnostics)

# Hamiltonian pieces (Hermitian)
# Note: adding constants (like zero-point ½ħω) would shift energies uniformly
# but not dynamics; we use the standard JC choice below.
H_c   = wc * num                          # Counter-Complexity energy operator
H_q   = 0.5 * wq * sz                     # Complexity energy operator
H_int = g  * (ad*sm + a*sp)               # Interaction operator
H     = H_c + H_q + H_int                 # Total Hamiltonian

# =============================================================================
# 3) INITIAL STATE (simple energy-exchange scenario)
# =============================================================================
# Start with one photon in the cavity and the qubit in |g⟩.
# On resonance, JC causes Rabi oscillations between |1,g⟩ and |0,e⟩.
psi0 = tensor(basis(N, 1), basis(2, 0))

# Closed evolution (no baths here — Part I focuses on the canonical split)
res = mesolve(H, psi0, tlist, c_ops=[], e_ops=[])
states = res.states

# =============================================================================
# 4) ENERGY DECOMPOSITION & DIAGNOSTICS (ties to “Energy Decomposition” section)
# =============================================================================
E_C_list    = [expect(H_q,   s) for s in states]  # E_complexity    (qubit sector)
E_CC_list   = [expect(H_c,   s) for s in states]  # E_counter-comp. (field sector)
E_int_list  = [expect(H_int, s) for s in states]  # E_interaction
E_tot_list  = [expect(H,     s) for s in states]  # E_total

# Extra signal for intuition: populations in each subsystem
n_cav_list  = [expect(num, s) for s in states]      # ⟨a†a⟩ (photon number)
p_exc_list  = [expect(P_e, s) for s in states]      # Prob(qubit excited)

# Energy-balance check (should be machine-precision small)
E_balance_err = np.max(np.abs(np.array(E_tot_list)
                              - (np.array(E_C_list) + np.array(E_CC_list) + np.array(E_int_list))))
print(f"Max energy-balance error: {E_balance_err:.3e}")

# =============================================================================
# 5) SAVE CSV (for GitHub preview / reproducibility)
# =============================================================================
csv_path = "jc_energy_timeseries.csv"
np.savetxt(
    csv_path,
    np.column_stack([
        tlist,
        E_C_list, E_CC_list, E_int_list, E_tot_list,
        n_cav_list, p_exc_list
    ]),
    delimiter=",",
    header="t,E_C,E_CC,E_int,E_tot,photon_number,p_qubit_excited",
    comments=""
)
print(f"Saved timeseries CSV -> {csv_path}")

# =============================================================================
# 6) PLOT & SAVE PNG (and show if interactive backend is available)
# =============================================================================
fig = plt.figure(figsize=(8, 7), constrained_layout=True)
gs = gridspec.GridSpec(2, 1, figure=fig)

# (a) Energies
ax = fig.add_subplot(gs[0, 0])
ax.plot(tlist, E_C_list,    label="E_C (complexity: qubit)")
ax.plot(tlist, E_CC_list,   label="E_\\bar{C} (counter-complexity: cavity)")
ax.plot(tlist, E_int_list,  label="E_int (interaction)")
ax.plot(tlist, E_tot_list,  "--", label="E_tot")
ax.set_ylabel("Energy (arb.)")
ax.set_title("JC Energy Decomposition — Complexity / Counter-Complexity Split")
ax.legend(loc="best")

# (b) Populations (for intuition: where the energy is sloshing)
ax = fig.add_subplot(gs[1, 0])
ax.plot(tlist, n_cav_list, label="⟨a†a⟩ (cavity photons)")
ax.plot(tlist, p_exc_list, label="P_e (qubit excited)")
ax.set_xlabel("time")
ax.set_ylabel("Population / Probability")
ax.legend(loc="best")

png_path = "jc_energy_mapping.png"
fig.savefig(png_path, dpi=180, bbox_inches="tight")
print(f"Saved plot PNG -> {png_path}")

# Try to show if a GUI backend is active; ignore if headless.
try:
    current_backend = matplotlib.get_backend().lower()
    if current_backend not in ("agg", "pdf", "svg", "svgcairo", "ps", "cairo"):
        plt.show()
except Exception:
    pass
finally:
    plt.close(fig)
