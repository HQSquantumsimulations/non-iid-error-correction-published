"""
This code plots logical error rates vs code sizes for a fixed per-qubit error rate below threshold
Simulates non-rotated XZZX code + single-qubit depolarizing noise
"""

import matplotlib.pyplot as plt
import json
from models.correlatednoise.nonrotatedplanarcode.generic import appcorrelated
from models.correlatednoise.nonrotatedplanarcode.generic import CorrelatedXZErrorModel
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarCodeXZ
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarMWPMDecoderIndependent
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarMWPMDecoderIndependentDeg


# set max_runs for each probability
max_runs = 5000

# set parameters of noise
# set physical error probabilities
error_probability = 0.1
# single-qubit gate error probability is  p1 = xi * error_probability
xi = 1.0

# initialize models
# initialize codes
codes = [PlanarCodeXZ(*size) for size in [(3, 3), (7, 7), (11, 11), (15, 15), (19, 19)]]

# initialize error models
error_model = CorrelatedXZErrorModel()

# initialize decoders
decoder = PlanarMWPMDecoderIndependent()
decoder_deg = PlanarMWPMDecoderIndependentDeg()

# run simulations
# Manhattan distance, no degeneracy:
data_independent = [appcorrelated.run(code, error_model, decoder, xi * error_probability, 1 / 2 * (1 - ((1 - 2 * error_probability) / (1 - 2 * xi * error_probability)) ** (1 / 4)), max_runs=max_runs)
                    for code in codes]
# Manhattan distance, wit degeneracy:
data_independent_deg = [appcorrelated.run(code, error_model, decoder_deg, xi * error_probability, 1 / 2 * (1 - ((1 - 2 * error_probability) / (1 - 2 * xi * error_probability)) ** (1 / 4)), max_runs=max_runs)
                        for code in codes]

# prepare code to x,y map
curve_independent = list()
for run in data_independent:
    curve_independent.append((run['n_k_d'][2], run['logical_failure_rate']))
# prepare code to x,y map
curve_independent_deg = list()
for run in data_independent_deg:
    curve_independent_deg.append((run['n_k_d'][2], run['logical_failure_rate']))

# plot data
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(*zip(*curve_independent), color='red', marker='D', linestyle='dashed',
    linewidth=4, markersize=18, label='Manhattan')
ax.plot(*zip(*curve_independent_deg), color='blue', marker='s', linestyle='dashed',
    linewidth=4, markersize=18, label='Manhattan + Deg')
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.legend(loc='lower left', prop={'size': 26})
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.xlabel('Code distance $\it{d}$', size=34)
plt.ylabel('Logical failure rate', size=34)
#plt.xlim(3, 30)
plt.yscale('log')
plt.show()
fig.savefig('./Fig_12a_quantum.pdf', format='pdf', bbox_inches='tight')
