import functools
import numpy as np
from qecsim import paulitools as pt
from qecsim.model import cli_description
from qecsim.models.generic import SimpleErrorModel
from qecsim.models.generic import DepolarizingErrorModel

@cli_description('Depolarizing single-qubit error + XZ error')
class CorrelatedXZErrorModel(SimpleErrorModel):
    """
    Implements a depolarizing single-qubit + 2-qubit XZ error model.
    """

    def generate(self, code, error_probability_1, error_probability, rng=None):
        """
        Generates single-qubit errors (depolarizing model) and two-qubit errors (XZ correlations)
        """

        n = code.size[0]
        m = code.size[1]
        d = n * m - 1
        step = code.n_k_d[0]
        # Probability of single-qubit error gate:
        error_probability_1 = error_probability_1
        # Probability of a two-qubit error gate. Qubits on the boundary have less nearest neighbours, hence the
        # effect of two-qubit errors overall reduced. To simulate correctly the effect of two-qubit noise we multiply
        # the two-qubit error probability by factor beta > 1, which becomes close to 1 for large codes.
        error_probability = error_probability * (n * m + (n - 1) * (m - 1)) / (4 * 0.25 + (2 * n + 2 * m - 8) * 0.5 + n * m + (n - 1) * (m - 1) - 2 * n - 2 * m + 4)
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n_k_d[0]
        # generate single-qubit errors
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=self.probability_distribution(error_probability_1)
        ))
        total_error = pt.pauli_to_bsf(error_pauli)
        # generate two-qubit errors
        # for two-qubit errors, the location of the qubit matters because edge qubits only have neibhours from one side
        for q in range(0, d - m - 1):
            i = q // m
            k = q % m
            col = k
            rnd = np.random.uniform(0, 1)
            if rnd < error_probability / 2:
                # multiply qubit #q by X and lower-left by Z
                if col > 0:
                    total_error[q] = (total_error[q] + 1) % 2
                    total_error[q + step] = total_error[q + step]
                    total_error[d + (m - 1) * i + k] = total_error[d + (m - 1) * i + k]
                    total_error[d + (m - 1) * i + k + step] = (total_error[d + (m - 1) * i + k + step] + 1) % 2
            elif error_probability / 2 < rnd < error_probability:
                # multiply qubit #q by Z and lower-left by X
                if col > 0:
                    total_error[q] = total_error[q]
                    total_error[q + step] = (total_error[q + step] + 1) % 2
                    total_error[d + (m - 1) * i + k] = (total_error[d + (m - 1) * i + k] + 1) % 2
                    total_error[d + (m - 1) * i + k + step] = total_error[d + (m - 1) * i + k + step]
        for q in range(0, d - m - 1):
            i = q // m
            k = q % m
            col = k
            rnd = np.random.uniform(0, 1)
            if rnd < error_probability / 2:
                # multiply qubit #q by X and lower-right by Z
                if col < m - 1:
                    total_error[q] = (total_error[q] + 1) % 2
                    total_error[q + step] = total_error[q + step]
                    total_error[d + (m - 1) * i + k + 1] = total_error[d + (m - 1) * i + k + 1]
                    total_error[d + (m - 1) * i + k + 1 + step] = (total_error[d + (m - 1) * i + k + 1 + step] + 1) % 2
            elif error_probability / 2 < rnd < error_probability:
                # multiply qubit #q by Z and lower-right by X
                if col < m - 1:
                    total_error[q] = total_error[q]
                    total_error[q + step] = (total_error[q + step] + 1) % 2
                    total_error[d + (m - 1) * i + k + 1] = (total_error[d + (m - 1) * i + k + 1] + 1) % 2
                    total_error[d + (m - 1) * i + k + 1 + step] = total_error[d + (m - 1) * i + k + 1 + step]
        for q in range(m, d):
            i = q // m
            k = q % m
            col = k
            rnd = np.random.uniform(0, 1)
            if rnd < error_probability / 2:
                # multiply qubit #q by X and upper-left by Z
                if col > 0:
                    total_error[q] = (total_error[q] + 1) % 2
                    total_error[q + step] = total_error[q + step]
                    total_error[d + (m - 1) * (i - 1) + k] = total_error[d + (m - 1) * (i - 1) + k]
                    total_error[d + (m - 1) * (i - 1) + k + step] = (total_error[d + (m - 1) * (i - 1) + k + step] + 1) % 2
            elif error_probability / 2 < rnd < error_probability:
                # multiply qubit #q by Z and upper-left by X
                if col > 0:
                    total_error[q] = total_error[q]
                    total_error[q + step] = (total_error[q + step] + 1) % 2
                    total_error[d + (m - 1) * (i - 1) + k] = (total_error[d + (m - 1) * (i - 1) + k] + 1) % 2
                    total_error[d + (m - 1) * (i - 1) + k + step] = total_error[d + (m - 1) * (i - 1) + k + step]
        for q in range(m, d):
            i = q // m
            k = q % m
            col = k
            rnd = np.random.uniform(0, 1)
            if rnd < error_probability / 2:
                # multiply qubit #q by X and upper-right by Z
                if col < m - 1:
                    total_error[q] = (total_error[q] + 1) % 2
                    total_error[q + step] = total_error[q + step]
                    total_error[d + (m - 1) * (i - 1) + k + 1] = total_error[d + (m - 1) * (i - 1) + k + 1]
                    total_error[d + (m - 1) * (i - 1) + k + 1 + step] = (total_error[d + (m - 1) * (i - 1) + k + 1 + step] + 1) % 2
            elif error_probability / 2 < rnd < error_probability:
                # multiply qubit #q by Z and upper-right by X
                if col < m - 1:
                    total_error[q] = total_error[q]
                    total_error[q + step] = (total_error[q + step] + 1) % 2
                    total_error[d + (m - 1) * (i - 1) + k + 1] = (total_error[d + (m - 1) * (i - 1) + k + 1] + 1) % 2
                    total_error[d + (m - 1) * (i - 1) + k + 1 + step] = total_error[d + (m - 1) * (i - 1) + k + 1 + step]
        return total_error

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = p_y = p_z = probability / 3
        return 1 - sum((p_x, p_y, p_z)), p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Depolarizing 1-qubit error + 2-qubit XZ error'

@cli_description('Depolarizing single-qubit error + XX error')
class CorrelatedXXErrorModel(SimpleErrorModel):
    """
    Implements a depolarizing single-qubit + 2-qubit XX error model.
    """

    def generate(self, code, error_probability_1, error_probability, rng=None):
        """
        Generates single-qubit errors (depolarizing model) and two-qubit errors (XX correlations)
        """

        n = code.size[0]
        m = code.size[1]
        d = n * m - 1
        step = code.n_k_d[0]
        # Probability of single-qubit error gate:
        error_probability_1 = error_probability_1
        # Probability of a two-qubit error gate. Qubits on the boundary have less nearest neighbours, hence the
        # effect of two-qubit errors overall reduced. To simulate correctly the effect of two-qubit noise we multiply
        # the two-qubit error probability by factor beta > 1, which becomes close to 1 for large codes.
        error_probability = error_probability * (n * m + (n - 1) * (m - 1)) / (
                    4 * 0.25 + (2 * n + 2 * m - 8) * 0.5 + n * m + (n - 1) * (m - 1) - 2 * n - 2 * m + 4)
        #  Generate single-qubit errors
        rng = np.random.default_rng()  # if rng is None else rng
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=self.probability_distribution(error_probability_1)
        ))
        total_error = pt.pauli_to_bsf(error_pauli)
        # Generate two-qubit errors
        # First, a loop over all primal qubits
        for q in range(0, d - m - 1):
            i = q // m
            k = q % m
            col = k
            # Error between a primal qubit q and bottom-left from it
            qubit_1 = q
            qubit_2 = d + (m - 1) * i + k
            if col > 0:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)
            # Error between a primal qubit q and bottom-right from it
            qubit_1 = q
            qubit_2 = d + (m - 1) * i + k + 1
            if col < m - 1:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)
        # a loop over all dual qubits
        for q in range(m, d):
            i = q // m
            k = q % m
            col = k
            qubit_1 = q
            qubit_2 = d + (m - 1) * (i - 1) + k
            if col > 0:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)
            qubit_1 = q
            qubit_2 = d + (m - 1) * (i - 1) + k + 1
            if col < m - 1:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)
        return total_error

    def two_qubit_error_generator(self, qubit_1_x, qubit_1_z, qubit_2_x, qubit_2_z, error_probability):
        rnd = np.random.uniform(0, 1)
        if (rnd < error_probability):
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        return qubit_1_x, qubit_1_z, qubit_2_x, qubit_2_z

    def mult_by_x(self, qubit_x, qubit_z):
        qubit_x = (qubit_x + 1) % 2
        qubit_z = qubit_z
        return qubit_x, qubit_z

    @functools.lru_cache()
    def probability_distribution(self, probability):
        p_x = p_y = p_z = probability / 3
        return 1 - sum((p_x, p_y, p_z)), p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Depolarizing 1-qubit error + 2-qubit XX error'

@cli_description('Depolarizing single-qubit error + depolarizing 2-qubit error')
class CorrelatedDepolarizingErrorModel(SimpleErrorModel):
    """
    Implements depolarizing single-qubit + depolarizing 2-qubit errors.
    """

    def generate(self, code, error_probability_1, error_probability, rng=None):
        """
        Generates single-qubit errors (depolarizing model) and two-qubit errors (depolarizing model)
        """

        n = code.size[0]
        m = code.size[1]
        d = n * m - 1
        step = code.n_k_d[0]

        # Probability of single-qubit error gate:
        error_probability_1 = error_probability_1
        # Probability of a two-qubit error gate. Qubits on the boundary have less nearest neighbours, hence the
        # effect of two-qubit errors overall reduced. To simulate correctly the effect of two-qubit noise we multiply
        # the two-qubit error probability by factor beta > 1, which becomes close to 1 for large codes.
        error_probability = error_probability * (n * m + (n - 1) * (m - 1)) / (4 * 0.25 + (2 * n + 2 * m - 8) * 0.5 + n * m + (n - 1) * (m - 1) - 2 * n - 2 * m + 4)
        #  Generate single-qubit errors
        rng = np.random.default_rng() # if rng is None else rng
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=self.probability_distribution(error_probability_1)
        ))
        total_error = pt.pauli_to_bsf(error_pauli)
        #  Generate two-qubit errors
        for q in range(0, d - m - 1):
            i = q // m
            k = q % m
            col = k
            qubit_1 = q
            qubit_2 = d + (m - 1) * i + k
            if col > 0:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)

            qubit_1 = q
            qubit_2 = d + (m - 1) * i + k + 1
            if col < m - 1:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)

        for q in range(m, d):
            i = q // m
            k = q % m
            col = k
            qubit_1 = q
            qubit_2 = d + (m - 1) * (i - 1) + k
            if col > 0:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)
            qubit_1 = q
            qubit_2 = d + (m - 1) * (i - 1) + k + 1
            if col < m - 1:
                total_error[qubit_1], total_error[qubit_1 + step], total_error[qubit_2], total_error[
                    qubit_2 + step] = self.two_qubit_error_generator(total_error[qubit_1], total_error[qubit_1 + step],
                                                                     total_error[qubit_2], total_error[qubit_2 + step],
                                                                     error_probability)

        return total_error
    def two_qubit_error_generator(self, qubit_1_x, qubit_1_z, qubit_2_x, qubit_2_z, error_probability):
        rnd = np.random.uniform(0, 1)
        if rnd < error_probability / 9:
            # multiply qubit #q by X and lower-left by X
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        elif error_probability / 9 < rnd < 2 * error_probability / 9:
            # multiply qubit #q by Y and lower-left by Y
            qubit_1_x, qubit_1_z = self.mult_by_y(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_y(qubit_2_x, qubit_2_z)
        elif 2 * error_probability / 9 < rnd < 3 * error_probability / 9:
            # multiply qubit #q by Z and lower-left by Z
            qubit_1_x, qubit_1_z = self.mult_by_z(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_z(qubit_2_x, qubit_2_z)
        elif 3 * error_probability / 9 < rnd < 4 * error_probability / 9:
            # multiply qubit #q by X and lower-left by Y
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_y(qubit_2_x, qubit_2_z)
        elif 4 * error_probability / 9 < rnd < 5 * error_probability / 9:
            # multiply qubit #q by Y and lower-left by X
            qubit_1_x, qubit_1_z = self.mult_by_y(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        elif 5 * error_probability / 9 < rnd < 6 * error_probability / 9:
            # multiply qubit #q by X and lower-left by Z
            qubit_1_x, qubit_1_z = self.mult_by_x(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_z(qubit_2_x, qubit_2_z)
        elif 6 * error_probability / 9 < rnd < 7 * error_probability / 9:
            # multiply qubit #q by Z and lower-left by X
            qubit_1_x, qubit_1_z = self.mult_by_z(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_x(qubit_2_x, qubit_2_z)
        elif 7 * error_probability / 9 < rnd < 8 * error_probability / 9:
            # multiply qubit #q by Y and lower-left by Z
            qubit_1_x, qubit_1_z = self.mult_by_y(qubit_1_x, qubit_1_z)
            qubit_2_x, qubit_2_z = self.mult_by_z(qubit_2_x, qubit_2_z)
        elif 8 * error_probability / 9 < rnd < error_probability:
            # multiply qubit #q by Z and lower-left by Y
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
        p_x = p_y = p_z = probability / 3
        return 1 - sum((p_x, p_y, p_z)), p_x, p_y, p_z
    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Depolarizing 1-qubit error + depolarizing 2-qubit error'
