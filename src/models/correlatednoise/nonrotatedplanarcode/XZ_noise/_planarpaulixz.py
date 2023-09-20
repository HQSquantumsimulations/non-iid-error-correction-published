import numpy as np
from qecsim.models.planar import PlanarPauli

class PlanarPauliXZ(PlanarPauli):
    """
    Defines a Pauli operator on a planar lattice.
    Same as PlanarPauli, but plaquettes with plaquettes measuring 2 Z and 2 X Paulis
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new XZ planar Pauli.
        """
        super().__init__(code, bsf)
        self._code = code
        self._from_bsf(bsf)

    @property
    def code(self):
        """
        The planar XZ code.

        :rtype: PlanarCode
        """
        return self._code

    def plaquette(self, index):
        """
        Apply a plaquette operator at the given index.

        :param index: Index identifying the plaquette in the format (row, column).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: PlanarPauli
        :raises IndexError: If index is not a plaquette index.
        """
        r, c = index
        # check valid index
        if not self.code.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        # flip plaquette sites
        self.site('Z', (r - 1, c))  # North
        self.site('Z', (r + 1, c))  # South
        self.site('X', (r, c - 1))  # West
        self.site('X', (r, c + 1))  # East
        return self

    def path(self, a_index, b_index):
        """
        Apply the shortest taxi-cab path of operators between the plaquettes indexed by A and B.
        Same as method:path of class:PlanarPauli, but each path can contain both X and Z operators
        """
        # steps from A to B
        row_steps, col_steps = self.code.translation(a_index, b_index)
        # current index
        c_r, c_c = a_index
        while row_steps < 0:  # heading north
            # flip current then decrement row
            self.site('X', (c_r - 1, c_c))
            c_r -= 2
            row_steps += 1
        while row_steps > 0:  # heading south
            # flip current then increment row
            self.site('X', (c_r + 1, c_c))
            c_r += 2
            row_steps -= 1
        while col_steps < 0:  # heading west
            # flip current then decrement col
            self.site('Z', (c_r, c_c - 1))
            c_c -= 2
            col_steps += 1
        while col_steps > 0:  # heading east
            # flip current then increment col
            self.site('Z', (c_r, c_c + 1))
            c_c += 2
            col_steps -= 1
        return self
