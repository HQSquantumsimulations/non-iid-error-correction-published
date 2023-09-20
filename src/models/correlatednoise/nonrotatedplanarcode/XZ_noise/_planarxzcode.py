from qecsim.model import cli_description
from qecsim.models.planar import PlanarCode
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarPauliXZ

@cli_description('Planar XZZX code')
class PlanarCodeXZ(PlanarCode):
    """
    Implements a planar mixed boundary XZZX code defined by its lattice size.

    """

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Planar Pauli
        :rtype: PlanarPauli
        """
        return PlanarPauliXZ(self, bsf)
