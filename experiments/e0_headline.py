"""E0: produce the headline numbers used in docs/00_summary.md.

Writes results/data/e0_headline.json so the docs can be regenerated rather than
copy-pasted by hand.
"""

from __future__ import annotations

import math

import numpy as np

from _common import dump, header, rate_at, scenario

from qcdma_qkd import attack, budget, cvqkd, loop, mui, security


def main() -> dict:
    header("E0  Headline numbers")
    sc = scenario()
    cfg, cdma, bcfg, sec = sc["cv"], sc["cdma"], sc["budget"], sc["sec"]
    L = cdma.spreading_factor
    K = cdma.n_users - 1
    snap = {}

    # --- physical baseline ------------------------------------------------ #
    snap["rate_0km"] = rate_at(cfg, 0.0)["S_pos"]
    snap["rate_25km"] = rate_at(cfg, 25.0)["S_pos"]
    snap["max_distance_clean"] = cvqkd.max_distance(cfg)

    # --- MUI --------------------------------------------------------------- #
    kappa = mui.uniform_kappa(K, 1.0)
    xi_mean = mui.xi_block_mean(kappa, L)
    xi_req = mui.required_xi(1e-12, kappa, L)
    snap.update(L=L, n_users=cdma.n_users,
                xi_mui_mean=xi_mean, xi_mui_required=xi_req,
                xi_mui_ratio=xi_req / xi_mean,
                xi_relative_std=mui.xi_block_std(kappa, L) / xi_mean,
                xi_relative_std_theory=math.sqrt(2.0 / K),
                rate_25km_with_mui=rate_at(cfg, 25.0, xi_req)["S_pos"],
                max_distance_with_mui=cvqkd.max_distance(cfg, xi_req))

    # --- authentication ----------------------------------------------------- #
    n_symbols = 10 ** 8
    sym_rate = cdma.symbol_rate()
    i_ab25 = rate_at(cfg, 25.0, xi_req)["I_AB"]
    b = budget.round_budget(rate_at(cfg, 25.0, xi_req)["S_pos"] * n_symbols,
                            n_symbols, i_ab25, bcfg)
    snap.update(round_symbols=n_symbols,
                round_seconds=n_symbols / sym_rate,
                chip_rate=cdma.chip_rate,
                symbol_rate=sym_rate,
                key_bits_per_round=b["K"],
                auth_bits_per_round=b["a"],
                auth_messages_per_round=b["n_messages"],
                seed_bits=b["c"],
                overhead_fraction=(b["a"] + b["c"]) / b["K"],
                auth_per_message_bits=budget.auth_key_bits(bcfg)["per_message"])

    # --- K_0 ---------------------------------------------------------------- #
    k0 = budget.k0_budget(bcfg, n_symbols, i_ab25)
    snap.update(k0_required=k0["required"], k0_available=k0["available"],
                k0_fraction=k0["fraction_of_k0"])

    # --- code model: PRF + f=1 is the sample-model answer -------------------- #
    chosen = cdma.refresh_constraint(bcfg.seed_bits, eps_code=1e-12)
    snap.update(code_model=chosen["code_model"],
                code_security=chosen["security"],
                code_refresh_fraction=chosen["refresh_fraction"],
                code_refresh_satisfied=chosen["satisfied"],
                code_f_min_chosen=chosen["f_min"])

    # --- refresh constraint, both routes (for the IT alternative) ------------ #
    eps_target = 1e-12
    con_ext = attack.refresh_security_constraint(bcfg.seed_bits, L,
                                                 eps_code=eps_target)
    con_prf = attack.refresh_security_constraint(bcfg.seed_bits, L, model="prf")
    con_tail = attack.refresh_constraint_tail(bcfg.seed_bits, L, 1.0,
                                              eps_code=eps_target)
    snap.update(refresh_f_min_extractor=con_ext["f_min"],
                refresh_f_min_prf=con_prf["f_min"],
                refresh_f_min_tail=con_tail["f_min"],
                refresh_eps_code_target=eps_target,
                refresh_k_req_bits=con_ext["k_req"],
                refresh_max_known_chips=con_ext["max_known_chips"])
    print(f"  code model = {chosen['code_model']}, f = "
          f"{chosen['refresh_fraction']:g} -> {chosen['security']}")
    print(f"  (extractor alternative would need f >= {con_tail['f_min']:.4f})")

    # --- MitM --------------------------------------------------------------- #
    snap.update(mitm_unauth_detection=attack.man_in_the_middle(
                    128, authenticated=False).detection_probability,
                mitm_auth_detection=attack.man_in_the_middle(
                    128, authenticated=True).detection_probability,
                mitm_survival_after_2_rounds=attack.detection_after_k_rounds(128, 8, 2))

    # --- state preparation -------------------------------------------------- #
    sp = {Ln: attack.state_preparation_leak(1.0, cfg.V_A, Ln)["xi_prep"]
          for Ln in (256, 1024, 4096)}
    snap["xi_prep_at_g1"] = sp
    snap["max_tolerable_g_1pct_L1024"] = attack.max_tolerable_partial_knowledge(
        0.01 * cfg.excess_noise, cfg.V_A, 1024)

    # --- epsilon ------------------------------------------------------------ #
    epsb = security.from_config(sec)
    snap.update(eps_total=epsb.total(), eps_target=sec.eps_total,
                eps_pass=bool(epsb.total() <= sec.eps_total),
                eps_sequential_1e4=security.compose_sequential(1e-12, 10 ** 4))

    # --- loop --------------------------------------------------------------- #
    lr = loop.run_loop(10, 2 ** 20, cfg, cdma, bcfg, 25.0, n_symbols=n_symbols)
    snap["loop_auth_ok"] = all(r.auth_ok for r in lr["rounds"])
    snap["loop_rounds"] = len(lr["rounds"])

    w = max(len(k) for k in snap)
    for k, v in snap.items():
        print(f"  {k:<{w}} = {v}")
    return snap


if __name__ == "__main__":
    dump("e0_headline", main())
