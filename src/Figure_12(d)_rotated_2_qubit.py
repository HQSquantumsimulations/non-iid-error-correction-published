"""
This code plots logical error rates vs code sizes for a fixed per-qubit error rate below threshold
Simulates rotated CSS code + two-qubit (XX + ZZ) noise
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from qecsim import app
from qecsim.models.rotatedplanar import RotatedPlanarCode
from models.correlatednoise.rotatedplanarcode import CorrelatedErrorModel
from models.correlatednoise.rotatedplanarcode import RotatedPlanarSMWPMDecoderDeg
from qecsim.models.rotatedplanar import RotatedPlanarSMWPMDecoder

# set max_runs for each probability
max_runs = 7500

# set models
codes = [RotatedPlanarCode(*size) for size in [(3, 3), (7, 7), (11, 11), (15, 15), (19, 19)]]
error_model = CorrelatedErrorModel()
decoder = RotatedPlanarSMWPMDecoder()
decoder_deg = RotatedPlanarSMWPMDecoderDeg()

# set physical error probabilities
error_probability = 0.05

# run simulations
data_independent = [app.run(code, error_model, decoder, error_probability, max_runs=max_runs)
        for code in codes]
data_independent_deg = [app.run(code, error_model, decoder_deg, error_probability, max_runs=max_runs)
        for code in codes]

# prepare code to x,y map
curve_independent = list()
for run in data_independent:
    curve_independent.append((run['n_k_d'][2], run['logical_failure_rate']))
curve_independent_deg = list()
for run in data_independent_deg:
    curve_independent_deg.append((run['n_k_d'][2], run['logical_failure_rate']))

# plot data
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(*zip(*curve_independent), color='red', marker='D', linestyle='dashed',
    linewidth=4, markersize=18, label='Manhattan')
ax.plot(*zip(*curve_independent_deg), color='blue', marker='o', linestyle='dashed',
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
#plt.ylim(0.005, 0.1)
plt.yscale('log')
plt.show()
fig.savefig('./Fig_12d_quantum.pdf', format='pdf', bbox_inches='tight')
