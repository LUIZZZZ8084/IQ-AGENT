import math

import numpy as np
import pytest

from iq_agent import estatistica as est


def test_breakeven_depende_do_payout():
    assert est.breakeven(1.0) == pytest.approx(0.5)
    assert est.breakeven(0.85) == pytest.approx(0.5405, abs=1e-4)
    assert est.breakeven(0.70) == pytest.approx(0.5882, abs=1e-4)


def test_valor_esperado_zera_no_breakeven():
    assert est.valor_esperado(est.breakeven(0.85), 0.85) == pytest.approx(0.0, abs=1e-12)
    assert est.valor_esperado(0.5, 0.85) < 0
    assert est.valor_esperado(0.6, 0.85) > 0


def test_kelly():
    assert est.kelly(0.5, 0.85) == 0.0  # sem vantagem, não aposta
    assert est.kelly(0.6, 0.85) == pytest.approx((0.6 * 1.85 - 1) / 0.85)
    assert est.kelly(0.9, 0.0) == 0.0


def test_wilson_contem_a_taxa_e_estreita_com_mais_dados():
    z = est.z_unilateral(0.95)
    inf_pouco, sup_pouco = est.wilson(60, 100, z)
    inf_muito, sup_muito = est.wilson(600, 1000, z)
    assert inf_pouco < 0.6 < sup_pouco
    assert inf_pouco < inf_muito < 0.6 < sup_muito < sup_pouco
    assert est.wilson(0, 0, z) == (0.0, 1.0)


def test_wilson_vetorizado_bate_com_o_escalar():
    z = est.z_unilateral(0.95)
    v, n = np.array([0, 5, 60, 600]), np.array([0, 10, 100, 1000])
    esperado = [0.0] + [est.wilson(a, b, z)[0] for a, b in zip(v[1:], n[1:])]
    assert est.wilson_inferior_vetor(v, n, z) == pytest.approx(esperado)


def test_p_valor_binomial_exato():
    # P(X >= 8) com n=10 e p=0.5 = (45 + 10 + 1) / 1024
    assert est.p_valor_binomial(8, 10, 0.5) == pytest.approx(56 / 1024)
    assert est.p_valor_binomial(10, 10, 0.5) == pytest.approx(1 / 1024)
    assert est.p_valor_binomial(0, 10, 0.5) == 1.0
    assert est.p_valor_binomial(11, 10, 0.5) == 0.0
    assert est.p_valor_binomial(5, 0, 0.5) == 1.0


def test_p_valor_binomial_confere_com_soma_direta():
    n, p0, v = 137, 0.54, 85
    direto = sum(math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k) for k in range(v, n + 1))
    assert est.p_valor_binomial(v, n, p0) == pytest.approx(direto, rel=1e-9)


def test_p_valor_nao_estoura_com_amostra_grande():
    p = est.p_valor_binomial(6000, 10000, 0.54)
    assert 0.0 <= p < 1e-20


def test_benjamini_hochberg():
    q = est.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert len(est.benjamini_hochberg([])) == 0
    assert est.benjamini_hochberg([0.9, 0.8]).max() <= 1.0
