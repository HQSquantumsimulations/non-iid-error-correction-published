from qecsim.model import cli_description
from models.localnoise._planarpaulimmhh import PlanarPauliMMHH
from models.localnoise import LocalCode

@cli_description('Planar MMHH (rows INT >= 2, cols INT >= 2)')
class LocalCodeMMHH(LocalCode):
    """
          Implements a non-rotated planar MMHH code.
          Calls :class:PlanarPauliMMHH to apply local Clifford rotations according to the known single-qubit errors
    """

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'MMHH code {}x{}'.format(*self.size)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.
        """
        return PlanarPauliMMHH(self, self.qubit_error_probabilities(), bsf)
