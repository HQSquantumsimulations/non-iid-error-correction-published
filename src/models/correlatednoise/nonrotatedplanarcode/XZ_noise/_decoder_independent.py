import functools
import math
from scipy import special
import numpy as np
from qecsim.model import cli_description
from models.correlatednoise.nonrotatedplanarcode.generic._planarmwpmdecoder import PlanarMWPMDecoder

@cli_description('MWPMDecoder')
class PlanarMWPMDecoderIndependent(PlanarMWPMDecoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder.
    Degeneracy factor is not taken into account
    """

    @classmethod
    @functools.lru_cache(maxsize=2 ** 26)  # to handle up to 100x100 codes.
    def distance(cls, code, a_index, b_index, p1, p2, degeneracy):
        """Distance function weighted to prefer steps along rows and allow for degeneracy if specified."""
        steps_along_rows, steps_along_cols = code.translation(a_index, b_index)
        separation = abs(steps_along_rows) + abs(steps_along_cols)
        return separation

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Planar MWPM Independent'

@cli_description('MWPMDecoderDegeneracy')
class PlanarMWPMDecoderIndependentDeg(PlanarMWPMDecoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder.
    Code degeneracy is taken into account when calculating the distance function
    """

    @classmethod
    @functools.lru_cache(maxsize=2 ** 26)  # to handle up to 100x100 codes.
    def distance(cls, code, a_index, b_index, p1, p2, degeneracy):
        """Distance function weighted to prefer steps along rows and allow for degeneracy if specified."""
        steps_along_rows, steps_along_cols = code.translation(a_index, b_index)
        separation = abs(steps_along_rows) + abs(steps_along_cols)
        # degeneracy = cls.degeneracy_term(abs(steps_along_rows), abs(steps_along_cols)) if degeneracy else 0
        a, b = np.sort([abs(steps_along_rows), abs(steps_along_cols)])
        degeneracy_simp = math.log(special.binom(a + b, a))
        return separation - degeneracy_simp

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Planar MWPM Independent Degenerate'
