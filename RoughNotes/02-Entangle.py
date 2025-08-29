#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entanglement under Env-mediated Dephasing (QuTiP)
-------------------------------------------------
This script corresponds to the notes:

- Part I (Complexity / Counter-Complexity):
  We split the Hamiltonian into "local/complexity" and "interaction" parts and
  verify the energy bookkeeping E_tot = E_loc + E_int (analog of
  E_total = E_complexity + E_counter-complexity + E_interaction in your notation).
  Here with two qubits, "counter-complexity" is implicit; we separate "local"
  vs "interaction" energies, which mirrors the decomposition logic.

- Part II (Entanglement & Environmental Mediation):
  We introduce a simple scalar environment proxy A_e(t) that increases with
  drive and relaxes over time. This acts as a *mediator term* by raising the
  dephasing rate gamma_phi(t), implementing our "environmental coupling"
  correction. We then monitor concurrence(t), <Z1 Z2>(t), and energy components.

Outputs:
  * PNG plot: entanglement_env_proxy.png
  * CSV data: entanglement_timeseries.csv

Requirements:
  pip install qutip matplotlib numpy
"""

import numpy as np
from qutip import (
    basis, tensor, qeye, sigmax, sigmay, sigmaz, sigmam,
    mesolve, expect, concurrence
)

# ----------------------------- Plot backend handling --------------------------
# If running headless (no GUI), we still want to save figures.
import matplotlib
try:
    # Try a GUI backend first; if it fails, fall back to Agg (headless).
    matplotlib.get_backend()
    # Some environments report a GUI backend but cannot show windows—handle later.
except Exception:
    pass
# Force Agg to be safe in headless; we will try to .show() and if it errors, we ignore.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# 1) PARAMETERS (connects to "Engineering Principles" in Part II)
# =============================================================================
# Local qubit "energies" (analogous to omega_complexity in your mapping)
w1, w2   = 1.0, 1.0

# Exchange coupling J generates entanglement.
# In Part II terms, this is the "direct" interaction energy contribution.
J        = 0.20

# Local drives inject "complexity" energy (pumping the local sectors).
# Increasing g_drive1 also increases the environment proxy via the mediator.
g_drive1 = 0.15
g_drive2 = 0.00

# Baseline noise rates (T2* dephasing, T1 relaxation).
# In your notes: gamma_phi0 ~ baseline decoherence; gamma_10 ~ baseline T1^-1.
gamma_phi0 = 0.01
gamma_10   = 0.005

# Mediator strength: how strongly the environment proxy A_e(t) raises dephasing.
# This implements the "environmental correction" you quantify (e.g., ~28% contrib).
k_env      = 0.15

# Environment proxy dynamics: dA_e/dt = -decay_env * A_e + |drive1| + |drive2|
# (Phenomenological: represents cavity photons, TLS, low-f noise, etc.)
decay_env  = 0.05

# Time grid for the simulation
tlist      = np.linspace(0, 50, 400)  # seconds (arbitrary units)

# =============================================================================
# 2) OPERATORS & HAMILTONIANS (connects to Part I energy decomposition)
# =============================================================================
I2 = qeye(2)
sx1, sy1, sz1, sm1 = tensor(sigmax(), I2), tensor(sigmay(), I2), tensor(sigmaz(), I2), tensor(sigmam(), I2)
sx2, sy2, sz2, sm2 = tensor(I2, sigmax()), tensor(I2, sigmay()), tensor(I2, sigmaz()), tensor(I2, sigmam())

# Local (complexity) part: qubit "energies" + local drives
# - This is the analog of E_complexity in your mapping (per-subsystem energy).
H_loc = 0.5 * w1 * sz1 + 0.5 * w2 * sz2 + g_drive1 * sx1 + g_drive2 * sx2

# Interaction part: XY exchange (entangling)
# - This contributes the "interaction" energy E_int in your mapping.
H_int = J * (sx1 * sx2 + sy1 * sy2)

# Total Hamiltonian (no explicit bath here; bath enters via time-varying collapse ops)
H = H_loc + H_int

# =============================================================================
# 3) INITIAL STATE (operational: entanglement via exchange)
# =============================================================================
# Start in |0,1> so exchange generates non-trivial dynamics and entanglement.
psi0 = tensor(basis(2, 0), basis(2, 1))
rho = psi0 * psi0.dag()

# =============================================================================
# 4) ENVIRONMENT PROXY & TIME-DEPENDENT NOISE (Part II mediator)
# =============================================================================
# A_e(t) grows with drive and relaxes with decay_env.
# It feeds into gamma_phi(t) = gamma_phi0 * (1 + k_env * A_e(t)).
A_e = 0.0
A_e_history = []

states = []      # density matrices over time
times_eff = []   # actual integration points (t0->t1 intervals)

# We step between consecutive times t0 -> t1, using piecewise-constant rates on each interval.
for i in range(len(tlist) - 1):
    t0 = float(tlist[i])
    t1 = float(tlist[i + 1])
    dt = t1 - t0

    # ----- Env proxy update (phenomenological mediator) -----
    # dA/dt = -decay_env * A + |drive1| + |drive2|
    A_e = A_e + dt * (-(decay_env) * A_e + abs(g_drive1) + abs(g_drive2))
    A_e_history.append(A_e)

    # ----- Time-dependent noise from mediator -----
    # Dephasing increases with environment amplitude (your environmental coupling correction).
    gamma_phi = gamma_phi0 * (1.0 + k_env * A_e)

    # Keep T1 fixed here, but could also depend on A_e if desired:
    gamma_1 = gamma_10

    # Collapse operators for dephasing (Z) and relaxation (lowering)
    c_ops = [
        np.sqrt(max(gamma_phi, 0.0)) * sz1,
        np.sqrt(max(gamma_phi, 0.0)) * sz2,
        np.sqrt(max(gamma_1,   0.0)) * sm1,
        np.sqrt(max(gamma_1,   0.0)) * sm2,
    ]

    # Evolve over [t0, t1] with piecewise-constant rates
    res = mesolve(H, rho, [t0, t1], c_ops=c_ops, e_ops=[])
    rho = res.states[-1]
    states.append(rho)
    times_eff.append(t1)

# Ensure arrays align (states are at tlist[1:], so we plot against tlist[:-1] or times_eff)
times_eff = np.array(times_eff)

# =============================================================================
# 5) OBSERVABLES (ties to Part II validation metrics)
# =============================================================================
ZZ = sz1 * sz2

# Concurrence measures entanglement of a two-qubit mixed state.
conc_list = [concurrence(r) for r in states]

# Correlation <Z1 Z2>
zz_list   = [expect(ZZ, r) for r in states]

# Energy bookkeeping (Part I analog: E_total = E_loc + E_int)
E_loc = [expect(H_loc, r) for r in states]
E_int = [expect(H_int, r) for r in states]
E_tot = [expect(H,     r) for r in states]

# Summaries (sanity checks)
max_conc = float(np.max(conc_list))
bal_err  = float(np.max(np.abs(np.array(E_tot) - (np.array(E_loc) + np.array(E_int)))))
print(f"Max concurrence: {max_conc:.3f}")
print(f"Max energy-balance error: {bal_err:.3e}")
print(f"Final env amplitude proxy A_e(T): {A_e_history[-1]:.3f}")

# =============================================================================
# 6) SAVE CSV (so GitHub can render a table preview)
# =============================================================================
csv_path = "entanglement_timeseries.csv"
np.savetxt(
    csv_path,
    np.column_stack([times_eff, conc_list, zz_list, E_loc, E_int, E_tot, A_e_history]),
    delimiter=",",
    header="t,concurrence,ZZ,E_loc,E_int,E_tot,A_e",
    comments=""
)
print(f"Saved timeseries CSV -> {csv_path}")

# =============================================================================
# 7) PLOT & SAVE PNG (show if possible; always save)
# =============================================================================
fig = plt.figure(figsize=(8, 7), constrained_layout=True)  # <-- replaces tight_layout()
gs = gridspec.GridSpec(3, 1, figure=fig)

ax = fig.add_subplot(gs[0, 0])
ax.plot(times_eff, conc_list)
ax.set_ylabel("Concurrence")
ax.set_title("Entanglement under Env-mediated Dephasing (QuTiP)\n"
             "Notes mapping: mediator A_e(t) raises γφ(t) -> entanglement affected")

ax = fig.add_subplot(gs[1, 0])
ax.plot(times_eff, zz_list)
ax.set_ylabel(r"$\langle Z_1 Z_2 \rangle$")

ax = fig.add_subplot(gs[2, 0])
ax.plot(times_eff, E_loc, label="E_loc (complexity/local)")
ax.plot(times_eff, E_int, label="E_int (interaction)")
ax.plot(times_eff, E_tot, "--", label="E_tot")
ax.set_ylabel("Energy")
ax.set_xlabel("time")
ax.legend(loc="best")

png_path = "entanglement_env_proxy.png"
fig.savefig(png_path, dpi=180, bbox_inches="tight")
print(f"Saved plot PNG -> {png_path}")

# Only show if backend is interactive (prevents the Agg warning)
if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "svgcairo", "ps", "cairo"):
    plt.show()
plt.close(fig)

# Try to show if a GUI backend is actually usable; ignore errors in headless runs.
try:
    plt.show()
except Exception:
    pass
