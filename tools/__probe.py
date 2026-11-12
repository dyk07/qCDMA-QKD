
import numpy as np
from qcdma_qkd import budget, config
bcfg = config.BudgetConfig(); sec = config.SecurityConfig()
n = 10**8; iab = 0.482428
Kn = 0.024107 * n          # PE closed-loop truth
a_axu = 207.726; a_otp = None

tr = budget.round_traffic(n, iab, bcfg)
print("round_traffic: pe=%.4e recon=%.4e meta=%.1f total=%.4e per_symbol=%.4f"
      % (tr["pe"], tr["recon"], tr["meta"], tr["total"], tr["total"]/n))
print("K_n (PE) = %.1f" % Kn)
print()
print("=== section 9 numbers, PE denominator ===")
for tag, msg_mod, B in (("AXU default", "axu", 1<<20), ("AXU 2^18", "axu", 1<<18),
                        ("OTP default", "otp", 1<<20), ("OTP 2^18", "otp", 1<<18)):
    cfg2 = config.BudgetConfig(mac_model=msg_mod)
    nm = budget.round_message_count(n, iab, cfg2, bits_per_message=B)
    a = budget.auth_bits(nm, cfg2, sec.eps_auth)
    print("  %-12s msgs=%5d  a=%9.1f  a/K=%.6e" % (tag, nm, a, a/Kn))
print()
print("  key/transcript ratio = %.6f  -> 1/%.1f" % (0.024107/(tr["total"]/n), (tr["total"]/n)/0.024107))
print("  tag/fragment        = 128/%d -> 1/%.0f" % ((1<<20), (1<<20)/128))
print()
print("=== what B or tag makes a/K = 10%? ===")
for B in (1<<18, 1<<19, 1<<20, 1<<21):
    nm = budget.round_message_count(n, iab, config.BudgetConfig(mac_model="otp"), bits_per_message=B)
    a = budget.auth_bits(nm, config.BudgetConfig(mac_model="otp"), sec.eps_auth)
    print("  OTP B=2^%d: a/K=%.4f %s" % (int(np.log2(B)), a/Kn, "<-- 10%" if abs(a/Kn-0.10)<0.03 else ""))
lo, hi = 16, 512
for _ in range(200):
    mid = 0.5*(lo+hi)
    cfg3 = config.BudgetConfig(mac_model="axu", tag_bits=int(round(mid)))
    nm = budget.round_message_count(n, iab, cfg3, bits_per_message=1<<20)
    a = budget.auth_bits(nm, cfg3, sec.eps_auth)
    if a/Kn < 0.10: lo = mid
    else: hi = mid
print("  AXU tag needed for 10%%: %.1f bit" % (0.5*(lo+hi)))
lo, hi = 1<<12, 1<<21
for _ in range(200):
    mid = int(0.5*(lo+hi))
    cfg4 = config.BudgetConfig(mac_model="otp")
    nm = budget.round_message_count(n, iab, cfg4, bits_per_message=int(mid))
    a = budget.auth_bits(nm, cfg4, sec.eps_auth)
    if a/Kn > 0.10: lo = mid
    else: hi = mid
print("  OTP fragment B needed for 10%%: %.0f bit (<=)" % (0.5*(lo+hi)))
