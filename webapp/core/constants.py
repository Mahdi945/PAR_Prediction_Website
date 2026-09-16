"""Physical constants for the web application.

⚠️ MIRROR of ``src/constants.py``. The web app is deployed standalone (its own
``requirements_web.txt``, no access to ``src/``), so the value is repeated here
rather than imported. ``tests/test_mccree_factor.py`` asserts the two agree, and
also checks the literals used in the pipeline notebooks — so the copies cannot
drift apart silently.
"""
from __future__ import annotations

# ── The McCree relation ───────────────────────────────────────────────────────
#
#   PAR [µmol m⁻² s⁻¹]  =  GHI [W m⁻²]  ×  0.45  ×  4.57
#
#   GHI                    W/m² = J s⁻¹ m⁻²   (broadband shortwave)
#   PAR_ENERGY_FRACTION    0.45        share of solar energy in the 400–700 nm band
#   PAR_QUANTUM_EFFICACY   4.57 µmol/J photons per joule in that band (McCree, 1972)
#
#   J s⁻¹ m⁻² × (dimensionless) × µmol J⁻¹ = µmol m⁻² s⁻¹   ✓
#
# 0.45 × 4.57 = 2.0565 → 2.06
PAR_ENERGY_FRACTION = 0.45      # dimensionless
PAR_QUANTUM_EFFICACY = 4.57     # µmol/J
MCCREE_FACTOR = 2.06            # µmol m⁻² s⁻¹ per W m⁻²

# Seconds per hour — converts an hourly PAR rate into a light integral.
# DLI [mol m⁻² day⁻¹] = Σ_hours ( PAR [µmol m⁻² s⁻¹] × 3600 ) / 1e6
SECONDS_PER_HOUR = 3600
MICROMOL_PER_MOL = 1e6
