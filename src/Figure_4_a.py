"""
This code plots logical error rates vs code sizes for a fixed per-qubit error rate below threshold
Simulates CSS vs MMHH codes
"""

import numpy as np
import matplotlib.pyplot as plt
from qecsim import app
from qecsim.graphtools import blossom5

# import models for QECC simulations
from models.localnoise import PlanarCodeCSS
# import noise model
from models.localnoise import LocalErrorModel
# import QECC
from models.localnoise import LocalCodeMMHH
# import decoders
from qecsim.models.planar import PlanarMWPMDecoder

#  set max_runs for each probability
max_runs = 6000

# code sized to simulate
sizes = [(3, 3), (7, 7), (11, 11), (15, 15), (19, 19), (23, 23)]

# set physical parameters of Pauli noise, see Eq. (17) of the paper
mean = 0.5
std = 0.25
std_total = 0.5

# random seed for probability generation
seed_high = np.random.randint(1, 1000)
seed_med = np.random.randint(1, 1000)
seed_low = np.random.randint(1, 1000)
seed_nonuniform = np.random.randint(1, 1000)

# set to True to generate a code with non-uniform total error probability centered at p with sigma = p/2
nonuniform = False

# initialise QECCs
# Error rates are drawn from normal distribution at the code initialization step

# MMHH code:
codes_mmhh = [LocalCodeMMHH(*size, mean=mean, std=std, seed_h=seed_high, seed_m=seed_med, seed_l=seed_low, seed_n=seed_nonuniform, nonuniform=nonuniform, std_t=std_total) for size in sizes]
# CSS code:
codes_css = [PlanarCodeCSS(*size, mean=mean, std=std, seed_h=seed_high, seed_m=seed_med, seed_l=seed_low, seed_n=seed_nonuniform, nonuniform=nonuniform, std_t=std_total) for size in sizes]

# Initialise the error model
# Generate errors according to the probabilities produced by method:qubit_error_probabilities of classs:LocalCode
error_model = LocalErrorModel()

# Initialize decoders
decoder_mwpm = PlanarMWPMDecoder()

# Set error probability
error_probability = 0.1

# Run simulations
data_mmhh_mwpm = [app.run(code, error_model, decoder_mwpm, error_probability, max_runs=max_runs)
        for code in codes_mmhh]
data_css_mwpm = [app.run(code, error_model, decoder_mwpm, error_probability, max_runs=max_runs)
        for code in codes_css]

# prepare code to x,y map
curve_mmhh_mwpm = list()
curve_css_mwpm = list()
for run in data_mmhh_mwpm:
    curve_mmhh_mwpm.append((run['n_k_d'][2], run['logical_failure_rate']))
for run in data_css_mwpm:
    curve_css_mwpm.append((run['n_k_d'][2], run['logical_failure_rate']))

# plot data
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(*zip(*curve_mmhh_mwpm), color='red', marker='o', linestyle='dashed',
    linewidth=4, markersize=18, label='MMHH code')
ax.plot(*zip(*curve_css_mwpm), color='blue', marker='s', linestyle='solid',
    linewidth=4, markersize=18, label='CSS code')
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.legend(loc='lower left', prop={'size': 26})
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.xlabel('Code distance $\it{d}$', size=34)
plt.ylabel('Logical failure rate', size=34)
#plt.xlim(3, 17)
plt.yscale('log')
plt.show()
fig.savefig('./Fig_4a_quantum.pdf', format='pdf', bbox_inches='tight')

