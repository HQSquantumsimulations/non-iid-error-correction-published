import functools
import math
import numpy as np
from scipy import special

from qecsim.model import cli_description
from models.correlatednoise.nonrotatedplanarcode.generic._planarmwpmdecoder import PlanarMWPMDecoder

@cli_description('MWPMCorrelatations')
class PlanarMWPMDecoderCorrelated(PlanarMWPMDecoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder
    Two-qubit correlations are explicitly taken into account when calculating the distance function.
    """

    @classmethod
    @functools.lru_cache(maxsize=2 ** 26)  # to handle up to 100x100 codes.
    def distance(cls, code, a_index, b_index, p1, p2, degeneracy):
        """Distance function weighted to prefer steps along rows and allow for degeneracy if specified."""
        steps_along_rows, steps_along_cols = code.translation(a_index, b_index)
        a, b = np.sort([abs(steps_along_rows), abs(steps_along_cols)])
        probability = p1 ** (a + b) + (1 - (a + b) % 2) * p2 ** b
        separation = -math.log(probability)
        return separation

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'MWPM Decoder + code degeneracy'

@cli_description('MWPMCorrelationsAndDegeneracy')
class PlanarMWPMDecoderCorrelatedDeg(PlanarMWPMDecoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder
    Code degeneracy and two-qubit correlations are taken into account when calculating the distance function.
    """

    @classmethod
    @functools.lru_cache(maxsize=2 ** 26)  # to handle up to 100x100 codes.
    def distance(cls, code, a_index, b_index, p1, p2, degeneracy):
        """Distance function weighted to prefer steps along rows and allow for degeneracy if specified."""
        steps_along_rows, steps_along_cols = code.translation(a_index, b_index)
        a, b = np.sort([abs(steps_along_rows), abs(steps_along_cols)])
        probability = special.binom(a + b, a) * p1 ** (a + b) + (1 - (a + b) % 2) * special.binom(b, (b - a) / 2) * p2 ** b
        separation = -math.log(probability)
        return separation

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'MWPM Decoder + code degeneracy + explicit correlations term'