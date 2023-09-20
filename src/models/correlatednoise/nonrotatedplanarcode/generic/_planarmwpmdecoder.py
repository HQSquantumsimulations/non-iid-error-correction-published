import functools
import itertools
import math

from qecsim import graphtools as gt
from qecsim.model import Decoder, cli_description

@cli_description('MWPM')
class PlanarMWPMDecoder(Decoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder.
    """

    def __init__(self, degeneracy=True):
        """
        Initialise new planar decoder.

        :param degeneracy: Apply degeneracy term. (default=True)
        :type degeneracy: bool
        """
        self._degeneracy = bool(degeneracy)

    @classmethod
    @functools.lru_cache(maxsize=2 ** 28)  # to handle up to 100x100 codes.
    def distance(cls, code, a_index, b_index, p1, p2, degeneracy):
        """Distance function weighted to prefer steps along rows and allow for degeneracy if specified."""
        steps_along_rows, steps_along_cols = code.translation(a_index, b_index)
        max_steps = max(abs(steps_along_rows), abs(steps_along_cols))
        probability = p1 ** (abs(steps_along_rows) + abs(steps_along_cols)) + p2 ** max_steps
        separation = -math.log(probability)
        return separation

    def decode(self, code, syndrome, error_probability_1, error_probability, **kwargs):
        """See :meth:`qecsim.model.Decoder.decode`"""
        # prepare recovery
        recovery_pauli = code.new_pauli()
        # get syndrome indices
        syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
        # split indices into primal and dual
        primal_indices = [i for i in syndrome_indices if code.is_primal(i)]
        dual_indices = [i for i in syndrome_indices if code.is_dual(i)]
        # extra virual indices are deliberately well off-boundary to be separate from nearest virtual indices
        primal_extra_vindex = (-9, -10)
        dual_extra_vindex = (-10, -9)
        # for each type of indices and extra virtual index
        for indices, extra_vindex in (primal_indices, primal_extra_vindex), (dual_indices, dual_extra_vindex):
            # prepare graph
            graph = gt.SimpleGraph()
            vindices = set()
            # add weighted edges between nodes and virtual nodes
            for index in indices:
                vindex = code.virtual_plaquette_index(index)
                vindices.add(vindex)
                distance = self.distance(code, index, vindex, error_probability_1, error_probability, self._degeneracy)
                graph.add_edge(index, vindex, distance)
            # add extra virtual node if odd number of total nodes
            if (len(indices) + len(vindices)) % 2:
                vindices.add(extra_vindex)
            # add weighted edges to graph between all (non-virtual) nodes
            for a_index, b_index in itertools.combinations(indices, 2):
                distance = self.distance(code, a_index, b_index, error_probability_1, error_probability, self._degeneracy)
                graph.add_edge(a_index, b_index, distance)
            # add zero weight edges between all virtual nodes
            for a_index, b_index in itertools.combinations(vindices, 2):
                graph.add_edge(a_index, b_index, 0)
            # find MWPM edges {(a, b), (c, d), ...}
            mates = gt.mwpm(graph)
            # iterate edges
            for a_index, b_index in mates:
                # add path to recover
                recovery_pauli.path(a_index, b_index)
        # return recover as bsf
        return recovery_pauli.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Planar MWPM'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
