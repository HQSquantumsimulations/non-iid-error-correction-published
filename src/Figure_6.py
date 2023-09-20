"""
This code plots logical error rates vs code sizes for a fixed per-qubit error rate below threshold
"""

import matplotlib.pyplot as plt
from models.correlatednoise.nonrotatedplanarcode.generic import appcorrelated
from models.correlatednoise.nonrotatedplanarcode.generic import CorrelatedXZErrorModel
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarCodeXZ
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarMWPMDecoderIndependent
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarMWPMDecoderIndependentDeg
from models.correlatednoise.nonrotatedplanarcode.XZ_noise import PlanarMWPMDecoderCorrelatedDeg


# set parameters of noise
# set physical error probabilities
error_probability = 0.1
# single-qubit gate error probability is  p1 = xi * error_probability
xi = 0.25

# set max_runs for each probability
max_runs = 10000

# initialize models

# initialize codes
codes = [PlanarCodeXZ(*size) for size in [(3, 3), (9, 9), (15, 15), (21, 21)]]

# initialize error models
error_model = CorrelatedXZErrorModel()

# initialize decoders
decoder = PlanarMWPMDecoderIndependent()
decoder_deg = PlanarMWPMDecoderIndependentDeg()
decoder_correlated_deg = PlanarMWPMDecoderCorrelatedDeg()

# run simulations
# Manhattan distance, no degeneracy:
data_independent = [appcorrelated.run(code, error_model, decoder, xi * error_probability, 1 / 2 * (1 - ((1 - 2 * error_probability) / (1 - 2 * xi * error_probability)) ** (1 / 4)), max_runs=max_runs)
                    for code in codes]
# Manhattan distance, with degeneracy:
data_independent_deg = [appcorrelated.run(code, error_model, decoder_deg, xi * error_probability, 1 / 2 * (1 - ((1 - 2 * error_probability) / (1 - 2 * xi * error_probability)) ** (1 / 4)), max_runs=max_runs)
                        for code in codes]
# Manhattan distance + 2 qubit correlations +  degeneracy:
data_correlated_deg = [appcorrelated.run(code, error_model, decoder_correlated_deg, xi * error_probability, 1 / 2 * (1 - ((1 - 2 * error_probability) / (1 - 2 * xi * error_probability)) ** (1 / 4)), max_runs=max_runs)
                       for code in codes]

# prepare code to x,y map and print
curve_independent = list()
for run in data_independent:
    curve_independent.append((run['n_k_d'][2], run['logical_failure_rate']))
curve_independent_deg = list()
for run in data_independent_deg:
    curve_independent_deg.append((run['n_k_d'][2], run['logical_failure_rate']))
curve_correlated_deg = list()
for run in data_correlated_deg:
    curve_correlated_deg.append((run['n_k_d'][2], run['logical_failure_rate']))

# plot data
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(*zip(*curve_independent), color='red', marker='D', linestyle='dashed',
    linewidth=4, markersize=18, label='Manhattan')
ax.plot(*zip(*curve_independent_deg), color='blue', marker='s', linestyle='dashed',
    linewidth=4, markersize=18, label='Manhattan + Deg')
ax.plot(*zip(*curve_correlated_deg), color='green', marker='o', linestyle='dashed',
    linewidth=4, markersize=18, label='Manhattan + Deg + Corr')
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
fig.savefig('./Fig_6_quantum.pdf', format='pdf', bbox_inches='tight')