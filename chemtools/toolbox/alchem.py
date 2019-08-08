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
"""The Alchemical Analysis Module."""


import numpy as np

from chemtools.wrappers.molecule import Molecule


__all__ = ['AlchemicalTool']


class AlchemicalTool(object):
    """Class of Density Functional Theory (DFT) Based Descriptive Tools."""

    def __init__(self, molecule, points):
        r"""Initialize class using instance of `Molecule` and grid points.

        Parameters
        ----------
        molecule : Molecule
            An instance of `Molecule` class
        points : np.ndarray
            Grid points, given as a 2D array with 3 columns, used for calculating local properties.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Argument points should be a 2D array with 3 columns.')

        self._molecule = molecule
        self._points = points
        # compute density, gradient, hessian & kinetic energy density on grid
        self._density = self._molecule.compute_density(self._points)

    @classmethod
    def from_molecule(cls, molecule, points):
        r"""Initialize class using instance of `Molecule` and points.

        Parameters
        ----------
        molecule : Molecule
            An instance of `Molecule` class.
        points : np.ndarray
            The (npoints, 3) array of cartesian coordinates of points.

        """
        return cls(molecule, points)

    @classmethod
    def from_file(cls, fname, points):
        """Initialize class from file.

        Parameters
        ----------
        fname : str
            Path to molecule's files.
        points : np.ndarray
            Grid points, given as a 2D array with 3 columns, used for calculating local properties.
        """
        molecule = Molecule.from_file(fname)
        return cls(molecule, points)

    @property
    def first_derivative(self):
        r"""Molecular Electrostatic Potential.

        .. math::
           V \left(\mathbf{r}\right) = \sum_A \frac{Z_A}{\rvert \mathbf{R}_A - \mathbf{r} \lvert} -
             \int \frac{\rho \left(\mathbf{r}"\right)}{\rvert \mathbf{r}" -
             \mathbf{r} \lvert} d\mathbf{r}"
        """
        correct = self._molecule.numbers / 1.e-3
        return self._molecule.compute_esp(self._points + 1.e-3 / np.sqrt(3)) - correct
