import functools
import itertools
import logging
import math
from collections import OrderedDict
from scipy import special

import numpy as np

from qecsim import graphtools as gt
from qecsim import paulitools as pt
from qecsim.error import QecsimError
from qecsim.model import Decoder, DecoderFTP, cli_description
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarSMWPMDecoder
logger = logging.getLogger(__name__)


@cli_description('Symmetry MWPM ([eta] FLOAT >=0), degeneracy is accounted for')
class RotatedPlanarSMWPMDecoderDeg(RotatedPlanarSMWPMDecoder):

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        # params as (name, value, non-default-falsy-values)
        params = [('eta', self._eta, ()), ]
        params_text = ', '.join('{}={}'.format(k, v) for k, v, f in params if v or v in f)
        return 'Deg Rotated planar SMWPM' + (' ({})'.format(params_text) if params_text else '')

    @classmethod
    def _distance(cls, code, time_steps, a_node, b_node,
                  error_probability=None, measurement_error_probability=0.0, eta=0.5):

        (a_t, a_x, a_y), a_is_row = a_node
        (b_t, b_x, b_y), b_is_row = b_node
        a_x, a_y = (a_x, a_y) if a_is_row else reversed((a_x, a_y))
        b_x, b_y = (b_x, b_y) if b_is_row else reversed((b_x, b_y))
        box_width = abs(a_x - b_x)
        box_height = abs(a_y - b_y)

        if box_width >= box_height:
            delta_parallel = box_width - box_height
            delta_diagonal = box_height
        else:
            delta_parallel = (box_height - box_width) % 2
            delta_diagonal = box_height

        #diag_steps = min(box_width, box_height)
        #par_steps = max(box_width, box_height) - diag_steps
        #x = diag_steps + par_steps / 2
        #y = par_steps / 2

        x = delta_diagonal + delta_parallel / 2
        y = delta_parallel / 2
        distance = x + y
        degeneracy = math.log(special.binom(x + y, y))
        return distance - degeneracy
