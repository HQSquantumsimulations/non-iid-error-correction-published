from qecsim.model import cli_description
from qecsim.models.planar import PlanarPauli
from models.localnoise import LocalCode

@cli_description('Planar (rows INT >= 2, cols INT >= 2)')
class PlanarCodeCSS(LocalCode):
    """
          Implements a non-ratated planar CSS code.
    """

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'CSS code {}x{}'.format(*self.size)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.
        """
        return PlanarPauli(self, bsf)
