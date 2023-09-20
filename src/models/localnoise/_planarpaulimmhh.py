import numpy as np
from qecsim.models.planar import PlanarPauli

class PlanarPauliMMHH(PlanarPauli):
    """
    Defines a Pauli operator on a planar lattice.
    Same as class:PlanarPauli up to local Clifford transformations
    Plaquettes are defined according to local error probabilities generates by :class:'LocalCodeMMHH'
    For each qubit, we choose:
    - Pauli-high to north and south and Pauli-medium to east and west if (abs(row - col)) % 4 == 0
    - Pauli-medium to north and south and Pauli-high to east and west if  (abs(row - col)) % 4 == 2
    """

    def __init__(self, code, qubit_prob, bsf=None):
        """
        Initialise new planar Pauli.
        """
        super().__init__(code, bsf)
        self._code = code
        self._qubit_prob = qubit_prob
        self._from_bsf(bsf)

    @property
    def code(self):
        """
        The planar MMHH code.

        :rtype: PlanarCode
        """
        return self._code

    def plaquette(self, index):
        """
        Similar to method:plaquette of class:PlanarPauli, but Pauli operators
        of each stabilzer are locally permuted according to local error probabilities
        """
        r, c = index
        # check valid index
        if not self.code.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        # call sites for Paulis to the north and south of the plaquette
        op_v, op_h = self.qubit_paulis((r - 1, c))
        self.site(op_v, (r - 1, c))  # North
        op_v, op_h = self.qubit_paulis((r + 1, c))
        self.site(op_v, (r + 1, c))  # South
        # call sites for Paulis to the west and east of the plaquette
        op_v, op_h = self.qubit_paulis((r, c - 1))
        self.site(op_h, (r, c - 1))  # West
        op_v, op_h = self.qubit_paulis((r, c + 1))
        self.site(op_h, (r, c + 1))  # East
        return self

    def qubit_paulis(self, index):
        """
        Called by method:plaquette to transform stabilizers locally
        Chooses the layout of Paulis around each qubit. This method defines the structure of the code
        Current choice:
        - Pauli-high to north and south and Pauli-medium to east and west if (abs(row - col)) % 4 == 0
        - Pauli-medium to north and south and Pauli-high to east and west if  (abs(row - col)) % 4 == 2
        """
        qubits_errors = self._qubit_prob
        row, col = index
        # check that qubits are queried
        if not self._code.is_plaquette(index):
            # Sort Paulis from the least to the most probable
            pauli = ['X', 'Y', 'Z']
            # Check whether the queried qubit exists
            if row < qubits_errors.shape[1] and col < qubits_errors.shape[2]:
                # sort errors from the least probable to the most probable
                Pauli_prob = [qubits_errors[0, row, col], qubits_errors[1, row, col], qubits_errors[2, row, col]]
                pauli_sorted = [x for _, x in sorted(zip(Pauli_prob, pauli))]
            # If not, it is a virtual qubit, and we can choose any operators for it, has no effect
            else:
                pauli_sorted = pauli
            high_rate = pauli_sorted[2]
            medium_rate = pauli_sorted[1]
            low_rate = pauli_sorted[0]
            # Layout Paulis according to the desired code structure (CURRENT CHOICE IS SUCH THAT LOGICALS = H-M-H-M-...)
            # NOTE: need to experiment with the optimal code layout
            if (abs(row - col)) % 4 == 0:
                op_v = high_rate
                op_h = medium_rate
            elif (abs(row - col)) % 4 == 2:
                op_h = high_rate
                op_v = medium_rate
        return op_v, op_h

    def path(self, a_index, b_index):
        """
        Apply the shortest taxi-cab path of operators between the plaquettes indexed by A and B.
        Operators are chosen accordingly to transformed stabilizers
        """
        # steps from A to B
        row_steps, col_steps = self.code.translation(a_index, b_index)
        c_r, c_c = a_index
        while row_steps < 0:  # heading north
            # flip current then decrement row
            operator = self.qubit_paulis((c_r - 1, c_c))[1]
            self.site(operator, (c_r - 1, c_c))
            c_r -= 2
            row_steps += 1
        while row_steps > 0:  # heading south
            # flip current then increment row
            operator = self.qubit_paulis((c_r + 1, c_c))[1]
            self.site(operator, (c_r + 1, c_c))
            c_r += 2
            row_steps -= 1
        while col_steps < 0:  # heading west
            # flip current then decrement col
            operator = self.qubit_paulis((c_r, c_c - 1))[0]
            self.site(operator, (c_r, c_c - 1))
            c_c -= 2
            col_steps += 1
        while col_steps > 0:  # heading east
            # flip current then increment col
            operator = self.qubit_paulis((c_r, c_c + 1))[0]
            self.site(operator, (c_r, c_c + 1))
            c_c += 2
            col_steps -= 1
        return self

    def logical_x(self):
        """
        Apply a logical X operator, i.e. column of Paulis to the west and east from each qubit on
        horizontal-edge sites of primal lattice. Defined automatically from :meth:qubit_paulis
        """
        max_row, max_col = self.code.bounds
        for row in range(0, max_row + 1, 2):
            self.site(self.qubit_paulis((row, max_col))[1], (row, max_col))
        return self

    def logical_z(self):
        """
        Apply a logical Z operator, i.e. row of Paulis to the north and south from each qubit on
        vertical-edge sites of primal lattice. Defined automatically from :meth:qubit_paulis
        """
        max_row, max_col = self.code.bounds
        for col in range(0, max_col + 1, 2):
            self.site(self.qubit_paulis((max_row, col))[0], (max_row, col))
        return self
