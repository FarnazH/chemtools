# -*- coding: utf-8 -*-
# ChemTools is a collection of interpretive chemical tools for
# analyzing outputs of the quantum chemistry calculations.
#
# Copyright (C) 2016-2019 The ChemTools Development Team
#
# This file is part of ChemTools.
#
# ChemTools is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# ChemTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Test chemtools.conceptual.rational Module."""

import sympy as sp
from numpy.testing import assert_raises, assert_equal, assert_almost_equal
from chemtools.conceptual.rational import RationalGlobalTool


def make_symbolic_rational_model(a, b, c, d):
    """Return rational energy, energy derivative & grand potential expressions."""
    ne = sp.symbols('ne')
    energy = (a + b * ne) / (c + d * ne)
    expr = (lambda n: float(energy.subs({'ne': n})),
            lambda n, r: float(sp.diff(energy, 'ne', r).subs({'ne': n})),
            lambda n: float(energy.subs({'ne': n}) - n * sp.diff(energy, 'ne', 1).subs({'ne': n})))
    return expr


def make_analytical_rational_grand_derivatives(deriv):
    """Return analytical 3rd and 4th derivatives of rational grand potential w.r.t. N."""
    expr = (lambda n: -1.0/deriv(n, 2),
            lambda n: deriv(n, 3)/deriv(n, 2)**3)
    # d4omega = lambda n: dE(n, 4) / dE(n, 2)**4 - (3 * dE(n, 3)**2) / dE(n, 2)**5
    # d5omega = lambda n: (dE(n, 5)/dE(n, 2)**5 - (10 * dE(n, 3) * dE(n, 4))/dE(n, 2)**6 +
    #                      (15 * dE(n, 3)**3)/dE(n, 2)**7)
    return expr


def test_global_rational_raises():
    # check invalid energy values
    assert_raises(ValueError, RationalGlobalTool, {5.: 15.0, 6.: 16.5, 4.: 18.1})
    assert_raises(ValueError, RationalGlobalTool, {6.: -15.0, 7.: -16.5, 5.: -18.1})
    assert_raises(ValueError, RationalGlobalTool, {10: -15.0, 11: -14.5, 9: -16.0})
    assert_raises(ValueError, RationalGlobalTool, {8: -15.0, 9: -14.9, 7: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {8: -15.0, 9: -15.0, 7: -16.0})
    # check invalid N0
    assert_raises(ValueError, RationalGlobalTool, {0: -15.0, 1: -14.4, -1: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {0.3: -15.0, 1.3: -14.4, -0.7: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {0.98: -15.0, 1.98: -14.4, -0.02: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {-1.: -15.0, 0.: -14.9, -2.: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {-2: -15.0, -1: -14.9, -3: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {0.8: -15.0, 0.0: -14.9, 1.5: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {5.0: -15.0, 4.5: -14.9, 5.5: -14.0})
    assert_raises(ValueError, RationalGlobalTool, {4.0: -15.0, 2.0: -14.9, 6.0: -14.0})
    # check invalid N
    model = RationalGlobalTool({5.: 5.2, 6.: 4.8, 4.: 6.0})
    assert_raises(ValueError, model.energy, -0.005)
    assert_raises(ValueError, model.energy, -1.35)
    assert_raises(ValueError, model.energy, -2.45)
    assert_raises(ValueError, model.energy_derivative, -0.05, 1)
    assert_raises(ValueError, model.energy_derivative, -1.91, 2)
    # check invalid derivative order
    assert_raises(ValueError, model.energy_derivative, 5.0, 1.)
    assert_raises(ValueError, model.energy_derivative, 5.0, 0.2)
    assert_raises(ValueError, model.energy_derivative, 5.0, -1)
    assert_raises(ValueError, model.energy_derivative, 5.0, -3)
    assert_raises(ValueError, model.energy_derivative, 5, '1')
    assert_raises(ValueError, model.energy_derivative, 5, [1])
    assert_raises(ValueError, model.energy_derivative, 3, 1.1)


def test_global_rational_pnpp_energy():
    # E(N) = (0.5 - 2.2 N) / (1 + 0.7 N)
    energy, deriv, _ = make_symbolic_rational_model(0.5, -2.2, 1., 0.7)
    # Build rational global tool instance
    model = RationalGlobalTool({2.: -1.6250, 3.: -1.96774193, 1.: -1.0})
    # check parameters
    assert_almost_equal(model.n0, 2.0, decimal=6)
    assert_almost_equal(model.params[0], 0.5, decimal=6)
    assert_almost_equal(model.params[1], -2.2, decimal=6)
    assert_almost_equal(model.params[2], 0.7, decimal=6)
    # check energy values (expected values are computed symbolically)
    assert_almost_equal(model.energy(0), energy(0), decimal=6)
    assert_almost_equal(model.energy(1), energy(1), decimal=6)
    assert_almost_equal(model.energy(2), energy(2), decimal=6)
    assert_almost_equal(model.energy(3), energy(3), decimal=6)
    assert_almost_equal(model.energy(4), energy(4), decimal=6)
    assert_almost_equal(model.energy(5), energy(5), decimal=6)
    assert_almost_equal(model.energy(6), energy(6), decimal=6)
    assert_almost_equal(model.energy(1.5), energy(1.5), decimal=6)
    assert_almost_equal(model.energy(0.8), energy(0.8), decimal=6)
    # check energy derivatives (expected values are computed symbolically)
    assert_almost_equal(model.energy_derivative(0), deriv(0, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(1), deriv(1, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(2), deriv(2, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(3), deriv(3, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(4), deriv(4, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(5), deriv(5, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(6), deriv(6, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(1.5), deriv(1.5, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(0.8), deriv(0.8, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(1.5, 2), deriv(1.5, 2), decimal=6)
    assert_almost_equal(model.energy_derivative(0.8, 2), deriv(0.8, 2), decimal=6)
    assert_almost_equal(model.energy_derivative(1.1, 3), deriv(1.1, 3), decimal=6)
    assert_almost_equal(model.energy_derivative(2.5, 4), deriv(2.5, 4), decimal=6)
    assert_almost_equal(model.energy_derivative(0.65, 5), deriv(0.65, 5), decimal=5)
    assert_almost_equal(model.energy_derivative(1.90, 6), deriv(1.90, 6), decimal=6)
    assert_almost_equal(model.energy_derivative(3.20, 3), deriv(3.20, 3), decimal=6)
    assert_almost_equal(model.energy_derivative(4.05, 7), deriv(4.05, 7), decimal=6)


def test_global_rational_pnpp_energy_reactivity():
    # E(N) = (0.5 - 2.2 N) / (1 + 0.7 N)
    energy, deriv, _ = make_symbolic_rational_model(0.5, -2.2, 1., 0.7)
    # Build rational global tool instance
    model = RationalGlobalTool({2.: -1.6250, 3.: -1.96774193, 1.: -1.0})
    # check global descriptors (expected values are computed symbolically)
    assert_almost_equal(model.ip, energy(1.0) - energy(2.0), decimal=6)
    assert_almost_equal(model.ea, energy(2.0) - energy(3.0), decimal=6)
    assert_almost_equal(model.mu, deriv(2.0, 1), decimal=6)
    assert_almost_equal(model.eta, deriv(2.0, 2), decimal=6)
    assert_almost_equal(model.ionization_potential, energy(1.0) - energy(2.0), decimal=6)
    assert_almost_equal(model.electron_affinity, energy(2.0) - energy(3.0), decimal=6)
    assert_almost_equal(model.chemical_potential, deriv(2.0, 1), decimal=6)
    assert_almost_equal(model.chemical_hardness, deriv(2.0, 2), decimal=6)
    assert_almost_equal(model.electronegativity, -deriv(2.0, 1), decimal=6)
    assert_almost_equal(model.hyper_hardness(2), deriv(2.0, 3), decimal=6)
    assert_almost_equal(model.hyper_hardness(3), deriv(2.0, 4), decimal=6)
    assert_almost_equal(model.hyper_hardness(4), deriv(2.0, 5), decimal=6)
    assert_almost_equal(model.hyper_hardness(5), deriv(2.0, 6), decimal=6)
    assert_almost_equal(model.hyper_hardness(6), deriv(2.0, 7), decimal=6)
    assert_almost_equal(model.softness, 1.0 / deriv(2.0, 2), decimal=6)
    # check n_max and related descriptors (expected values are computed symbolically)
    assert_equal(model.n_max, float('inf'))
    assert_almost_equal(model.energy(model.n_max), -3.14285714, decimal=6)
    assert_almost_equal(model.energy_derivative(model.n_max), 0.0, decimal=6)
    assert_almost_equal(model.energy_derivative(model.n_max, 2), 0.0, decimal=6)
    assert_almost_equal(model.energy_derivative(model.n_max, 3), 0.0, decimal=6)
    assert_almost_equal(model.electrophilicity, 1.51785714, decimal=6)
    assert_almost_equal(model.nucleofugality, -1.17511520, decimal=6)
    assert_almost_equal(model.electrofugality, 2.14285714, decimal=6)


def test_global_rational_pnpp_grand_potential():
    # E(N) = (0.5 - 2.2 N) / (1 + 0.7 N)
    _, deriv, grand = make_symbolic_rational_model(0.5, -2.2, 1., 0.7)
    # Build rational global tool instance
    model = RationalGlobalTool({2.: -1.6250, 3.: -1.96774193, 1.: -1.0})
    # check grand potential (as a function of N)
    assert_almost_equal(model.grand_potential(1.), grand(1.), decimal=6)
    assert_almost_equal(model.grand_potential(2.), grand(2.0), decimal=6)
    assert_almost_equal(model.grand_potential(3.), grand(3.0), decimal=6)
    assert_almost_equal(model.grand_potential(2.78), grand(2.78), decimal=6)
    assert_almost_equal(model.grand_potential(5.2), grand(5.2), decimal=6)
    assert_almost_equal(model.grand_potential(0.), grand(0.), decimal=6)
    # assert_almost_equal(model.grand_potential(model.n_max), , decimal=6)
    # check grand potential derivative (as a function of N)
    assert_almost_equal(model.grand_potential_derivative(2.), -2.0, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(1.4, 1), -1.4, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(2.86), -2.86, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(2., 2), -3.87226890, decimal=6)
    # expected values based on derived formulas
    d2omega, d3omega = make_analytical_rational_grand_derivatives(deriv)
    assert_almost_equal(model.grand_potential_derivative(4.67, 1), -4.67, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(3.5, 2), d2omega(3.5), decimal=6)
    assert_almost_equal(model.grand_potential_derivative(4.1, 2), d2omega(4.1), decimal=6)
    assert_almost_equal(model.grand_potential_derivative(4.67, 2), d2omega(4.67), decimal=6)
    assert_almost_equal(model.grand_potential_derivative(2.9, 3), d3omega(2.9), decimal=5)
    assert_almost_equal(model.grand_potential_derivative(4.67, 3), d3omega(4.67), decimal=4)
    # assert_almost_equal(model.grand_potential_derivative(1.6, 4), d4omega(1.6), decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(2.92, 4), d4omega(2.92), decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(5.01, 5), d5omega(5.01), decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(4.101, 5), d5omega(4.101), decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(model.n_max, 4), decimal=6)
    # check mu to N conversion
    assert_almost_equal(model.convert_mu_to_n(-0.4427083333), 2., decimal=6)
    assert_almost_equal(model.convert_mu_to_n(-0.5799422391), 1.567, decimal=6)
    assert_almost_equal(model.convert_mu_to_n(-0.9515745573), 0.91, decimal=6)
    assert_almost_equal(model.convert_mu_to_n(-0.2641542934), 3.01, decimal=6)
    # check grand potential (as a function of mu)
    assert_almost_equal(model.grand_potential_mu(-0.125925925), -1.70370370, decimal=6)
    assert_almost_equal(model.grand_potential_mu(-0.442708333), -0.73958333, decimal=6)
    assert_almost_equal(model.grand_potential_mu(-0.232747054), -1.27423079, decimal=6)
    # check grand potential derivative (as a function of mu)
    assert_almost_equal(model.grand_potential_mu_derivative(deriv(5.81, 1), 1), -5.81, decimal=6)
    mu = deriv(4.67, 1)
    assert_almost_equal(model.grand_potential_mu_derivative(mu, 2), d2omega(4.67), decimal=5)
    # assert_almost_equal(domega_mu(dE(6.45, 1), 3), d3omega(6.45), decimal=6)
    # assert_almost_equal(domega_mu(dE(5.12, 1), 4), d4omega(5.12), decimal=6)


def test_global_rational_pnpp_grand_potential_reactivity():
    # E(N) = (0.5 - 2.2 N) / (1 + 0.7 N)
    n0, a0, a1, b1 = 2.0, 0.5, -2.2, 0.7
    # build global tool
    model = RationalGlobalTool({2.: -1.6250, 3.: -1.96774193, 1.: -1.0})
    # check hyper-softnesses
    expected = 3.0 * (1 + b1 * n0)**5 / (4 * b1 * (a1 - a0 * b1)**2)
    assert_almost_equal(model.hyper_softness(2), expected, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(2.0, 3), -expected, decimal=6)
    expected = -15 * (1 + b1 * n0)**7 / (8 * b1 * (a1 - a0 * b1)**3)
    assert_almost_equal(model.hyper_softness(3), expected, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(2.0, 4), -expected, decimal=6)


def test_global_rational_nnpp_energy():
    # E(N) = (-0.15 - 4.2 N) / (1 + 0.45 N)
    energy, deriv, _ = make_symbolic_rational_model(-0.15, -4.2, 1., 0.45)
    # build global tool
    model = RationalGlobalTool({6.5: -6.99363057, 7.5: -7.23428571, 5.5: -6.69064748})
    # check parameters
    assert_almost_equal(model.n0, 6.5, decimal=6)
    assert_almost_equal(model.params[0], -0.15, decimal=6)
    assert_almost_equal(model.params[1], -4.2, decimal=6)
    assert_almost_equal(model.params[2], 0.45, decimal=6)
    # check energy values (expected values are computed symbolically)
    assert_almost_equal(model.energy(6.5), energy(6.5), decimal=6)
    assert_almost_equal(model.energy(7.5), energy(7.5), decimal=6)
    assert_almost_equal(model.energy(5.5), energy(5.5), decimal=6)
    assert_almost_equal(model.energy(5.0), energy(5.0), decimal=6)
    assert_almost_equal(model.energy(8.0), energy(8.0), decimal=6)
    # check energy derivatives (expected values are computed symbolically)
    assert_almost_equal(model.energy_derivative(6.5), deriv(6.5, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(7.5), deriv(7.5, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(5.5), deriv(5.5, 1), decimal=6)
    assert_almost_equal(model.energy_derivative(4, 2), deriv(4.0, 2), decimal=6)
    assert_almost_equal(model.energy_derivative(10., 3), deriv(10., 3), decimal=6)
    assert_almost_equal(model.energy_derivative(9.5, 4), deriv(9.5, 4), decimal=6)


def test_global_rational_nnpp_energy_reactivity():
    # E(N) = (-0.15 - 4.2 N) / (1 + 0.45 N)
    energy, deriv, _ = make_symbolic_rational_model(-0.15, -4.2, 1., 0.45)
    # build global tool
    model = RationalGlobalTool({6.5: -6.99363057, 7.5: -7.23428571, 5.5: -6.69064748})
    # check global descriptors (expected values are computed symbolically)
    assert_almost_equal(model.ip, energy(5.5) - energy(6.5), decimal=6)
    assert_almost_equal(model.ea, energy(6.5) - energy(7.5), decimal=6)
    assert_almost_equal(model.mu, deriv(6.5, 1), decimal=6)
    assert_almost_equal(model.eta, deriv(6.5, 2), decimal=6)
    assert_almost_equal(model.ionization_potential, energy(5.5) - energy(6.5), decimal=6)
    assert_almost_equal(model.electron_affinity, energy(6.5) - energy(7.5), decimal=6)
    assert_almost_equal(model.chemical_potential, deriv(6.5, 1), decimal=6)
    assert_almost_equal(model.chemical_hardness, deriv(6.5, 2), decimal=6)
    assert_almost_equal(model.electronegativity, -deriv(6.5, 1), decimal=6)
    assert_almost_equal(model.hyper_hardness(2), deriv(6.5, 3), decimal=6)
    assert_almost_equal(model.hyper_hardness(3), deriv(6.5, 4), decimal=6)
    assert_almost_equal(model.hyper_hardness(4), deriv(6.5, 5), decimal=6)
    assert_almost_equal(model.hyper_hardness(5), deriv(6.5, 6), decimal=6)
    assert_almost_equal(model.hyper_hardness(6), deriv(6.5, 7), decimal=6)
    assert_almost_equal(model.softness, 1. / deriv(6.5, 2), decimal=6)
    # check n_max and related descriptors (expected values are computed symbolically)
    assert_equal(model.n_max, float('inf'))
    assert_almost_equal(model.energy(model.n_max), -9.33333333, decimal=6)
    assert_almost_equal(model.energy_derivative(model.n_max), 0.0, decimal=6)
    assert_almost_equal(model.energy_derivative(model.n_max, 2), 0.0, decimal=6)
    assert_almost_equal(model.energy_derivative(model.n_max, 3), 0.0, decimal=6)
    assert_almost_equal(model.electrophilicity, 2.33970276, decimal=6)
    assert_almost_equal(model.nucleofugality, -2.099047619, decimal=6)
    assert_almost_equal(model.electrofugality, 2.64268585, decimal=6)


def test_global_rational_nnpp_grand_potential():
    # E(N) = (-0.15 - 4.2 N) / (1 + 0.45 N)
    _, deriv, _ = make_symbolic_rational_model(-0.15, -4.2, 1., 0.45)
    # build global tool
    model = RationalGlobalTool({6.5: -6.99363057, 7.5: -7.23428571, 5.5: -6.69064748})
    # check grand potential (as a function of N)
    assert_almost_equal(model.grand_potential(6.5), -5.2500304, decimal=6)
    assert_almost_equal(model.grand_potential(7.91), -5.7468530, decimal=6)
    assert_almost_equal(model.grand_potential(0.), -0.15, decimal=6)
    # assert_almost_equal(model.grand_potential(model.n_max), , decimal=6)
    # check grand potential derivative (as a function of N)
    # expected values based on derived formulas
    d2omega, d3omega = make_analytical_rational_grand_derivatives(deriv)
    assert_almost_equal(model.grand_potential_derivative(6.5, 1), -6.5, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(7.1, 1), -7.1, decimal=6)
    assert_almost_equal(model.grand_potential_derivative(5.8, 2), d2omega(5.8), decimal=6)
    assert_almost_equal(model.grand_potential_derivative(0.0, 3), d3omega(0.0), decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(8.01, 4), d4omega(8.01), decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(6.901, 5), d5omega(6.901), decimal=6)
    # check mu to N conversion
    assert_almost_equal(model.convert_mu_to_n(-0.2682461763), 6.5, decimal=6)
    assert_almost_equal(model.convert_mu_to_n(-0.2345757894), 7.105, decimal=6)
    assert_almost_equal(model.convert_mu_to_n(-0.1956803972), 7.99, decimal=6)
    assert_almost_equal(model.convert_mu_to_n(-0.3568526811), 5.34, decimal=6)
    # check grand potential (as a function of mu)
    assert_almost_equal(model.grand_potential_mu(-0.26824617), -5.2500304, decimal=6)
    assert_almost_equal(model.grand_potential_mu(-0.19153203), -5.8048876, decimal=6)
    assert_almost_equal(model.grand_potential_mu(-0.20521256), -5.6965107, decimal=6)
    # assert_almost_equal(model.grand_potential_derivative(model.n_max, 4), , decimal=6)
    assert_almost_equal(model.grand_potential_mu(-0.268246176), -5.2500304, decimal=6)
    assert_almost_equal(model.grand_potential_mu(-0.198782625), -5.7468530, decimal=6)
    # check grand potential derivative (as a function of mu)
    assert_almost_equal(model.grand_potential_mu_derivative(deriv(6.301, 1), 1), -6.301, decimal=6)
    mu = deriv(5.55, 1)
    assert_almost_equal(model.grand_potential_mu_derivative(mu, 2), d2omega(5.55), decimal=6)
    mu = deriv(6.99, 1)
    assert_almost_equal(model.grand_potential_mu_derivative(mu, 3), d3omega(6.99), decimal=6)
    # assert_almost_equal(domega_mu(dE(7.1, 1), 4), d4omega(7.1), decimal=6)
    # assert_almost_equal(domega_mu(dE(7.6, 1), 5), d5omega(7.6), decimal=6)


def test_global_rational_nnpp_grand_potential_reactivity():
    # E(N) = (-0.15 - 4.2 N) / (1 + 0.45 N)
    n0, a0, a1, b1 = 6.5, -0.15, -4.2, 0.45
    # build global tool
    model = RationalGlobalTool({6.5: -6.99363057, 7.5: -7.23428571, 5.5: -6.69064748})
    # check hyper-softnesses
    expected = 3.0 * (1 + b1 * n0)**5 / (4 * b1 * (a1 - a0 * b1)**2)
    assert_almost_equal(model.hyper_softness(2), expected, decimal=5)
    assert_almost_equal(model.grand_potential_derivative(6.5, 3), -expected, decimal=5)
    expected = -15 * (1 + b1 * n0)**7 / (8 * b1 * (a1 - a0 * b1)**3)
    assert_almost_equal(model.hyper_softness(3), expected, decimal=4)
    assert_almost_equal(model.grand_potential_derivative(6.5, 4), -expected, decimal=4)
