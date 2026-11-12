
import numpy as np
from qcdma_qkd import mui, config, loop, cvqkd
sec = config.SecurityConfig(); cfg = config.CVQKDConfig()
bcfg = config.BudgetConfig(); cdma = config.CDMAConfig()
kappa = np.ones(7)
print("chi2 exact   =", mui.required_xi(1e-12, kappa, 1024))
print("L-M bound    =", mui.xi_block_bound(1e-12, kappa, 1024))
print("xi_block_exact =", mui.xi_block_exact(1e-12, kappa, 1024))
print()
for model in ("quantile", "bound"):
    kw = dict(cfg_cv=cfg, cfg_cdma=cdma, cfg_bud=bcfg, distance_km=25.0,
              eps_mui=sec.eps_mui, eps_pe=sec.eps_qkd, eps_sm=sec.eps_sm,
              eps_pa=sec.eps_pa, xi_mui_model=model)
    r = loop.simulate_round(0, b"s"*32, 10**8, **kw)
    print("%-9s xi_mui=%.5f xi_used=%.5f S=%.6f K=%.0f"
          % (model, r.xi_mui, r.extras["xi_used"], r.rate_per_symbol, r.k_bits))
