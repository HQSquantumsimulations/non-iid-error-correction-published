import abc
import functools

import numpy as np

from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description

@cli_description('Depolarizing error + depolarizing 2-qubit error')
class CorrelatedErrorModel(ErrorModel):

    def generate(self, code, error_probability, rng=None):

        # max index along x and y
        n = code.n_k_d[2] - 1
        m = code.n_k_d[2] - 1

        # number of qubits
        N = code.n_k_d[0]

        #  Generate single-qubit errors (there are none)
        rng = np.random.default_rng()  # if rng is None else rng
        n_qubits = code.n_k_d[0]
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=self.probability_distribution(0.0)
        ))
        total_error = pt.pauli_to_bsf(error_pauli)

        # Generate 2-qubit errors
        # Error probability p per qubit means approx p/4 per gate
        error_probability = error_probability / 4
        # loop over all qubits
        for q in range(0, N-1):
            row = q // (n + 1)
            col = q % (n + 1)
            # add errors between qubit q and one to the right from q
            if col < n:
                # generate a random number
                rnd = np.random.uniform(0, 1)
                # create XX interaction
                if(rnd <= error_probability / 2):
                    total_error[q] = (total_error[q] + 1) % 2
                    total_error[q + 1] = (total_error[q + 1] + 1) % 2
                # create ZZ interaction
                elif (error_probability / 2 < rnd <= error_probability):
                    total_error[q + N] = (total_error[q + N] + 1) % 2
                    total_error[q + 1 + N] = (total_error[q + 1 + N] + 1) % 2
            if row < n:
                # add errors between each qubit and one step up from q
                rnd = np.random.uniform(0, 1)
                # create XX interaction
                if (rnd <= error_probability / 2):
                    total_error[q] = (total_error[q] + 1) % 2
                    total_error[q + n + 1] = (total_error[q + n + 1] + 1) % 2
                # create ZZ interaction
                elif (error_probability / 2 < rnd <= error_probability):
                    total_error[q + N] = (total_error[q + N] + 1) % 2
                    total_error[q + n + 1 + N] = (total_error[q + n + 1 + N] + 1) % 2
        return total_error

    def two_qubit_error_generator(self, qubit_1_x, qubit_1_z, qubit_2_x, qubit_2_z, error_probability):
        rnd = np.random.uniform(0, 1)
        if (rnd < error_probability / 9):
            # multiply qubit #q by X and lower-left by X
            # print(['XX', q])
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        elif (error_probability / 9 < rnd < 2 * error_probability / 9):
            # multiply qubit #q by Y and lower-left by Y
            # print(['YY', q])
            qubit_1_x, qubit_1_z = self.mult_by_y(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_y(qubit_2_x, qubit_2_z)
        elif (2 * error_probability / 9 < rnd < 3 * error_probability / 9):
            # multiply qubit #q by Z and lower-left by Z
            # print(['ZZ', q])
            qubit_1_x, qubit_1_z = self.mult_by_z(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_z(qubit_2_x, qubit_2_z)
        elif (3 * error_probability / 9 < rnd < 4 * error_probability / 9):
            # multiply qubit #q by X and lower-left by Y
            # print(['XY', q])
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_y(qubit_2_x, qubit_2_z)
        elif (4 * error_probability / 9 < rnd < 5 * error_probability / 9):
            # multiply qubit #q by Y and lower-left by X
            # print(['YX', q])
            qubit_1_x, qubit_1_z = self.mult_by_y(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        elif (5 * error_probability / 9 < rnd < 6 * error_probability / 9):
            # multiply qubit #q by X and lower-left by Z
            # print(['XZ', q])
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_z(qubit_2_x, qubit_2_z)
        elif (6 * error_probability / 9 < rnd < 7 * error_probability / 9):
            # multiply qubit #q by Z and lower-left by X
            # print(['ZX', q])
            qubit_1_x, qubit_1_z = self.mult_by_z(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        elif (7 * error_probability / 9 < rnd < 8 * error_probability / 9):
            # multiply qubit #q by Y and lower-left by Z
            # print(['YZ', q])
            qubit_1_x, qubit_1_z = self.mult_by_y(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_z(qubit_2_x, qubit_2_z)
        elif (8 * error_probability / 9 < rnd < error_probability):
            # multiply qubit #q by Z and lower-left by Y
            # print(['ZY', q])
            qubit_1_x, qubit_1_z = self.mult_by_z(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_y(qubit_2_x, qubit_2_z)
        return qubit_1_x, qubit_1_z, qubit_2_x, qubit_2_z

    def mult_by_x(self, qubit_x, qubit_z):
        qubit_x = (qubit_x + 1) % 2
        qubit_z = qubit_z
        return qubit_x, qubit_z

    def mult_by_z(self, qubit_x, qubit_z):
        qubit_x = qubit_x
        qubit_z = (qubit_z + 1) % 2
        return qubit_x, qubit_z

    def mult_by_y(self, qubit_x, qubit_z):
        qubit_x = (qubit_x + 1) % 2
        qubit_z = (qubit_z + 1) % 2
        return qubit_x, qubit_z

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = p_y = p_z = probability / 3
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Correlated XZ'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)