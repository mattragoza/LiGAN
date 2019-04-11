import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
from math import exp
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')
seaborn.set_context('poster')

sys.path.append('.')
import generate

a = np.zeros((1, 1))
r = 1.0
p = np.linspace(-2, 2, 100)[:,np.newaxis]
d = generate.get_atom_density(a, r, p, 1.5)

fig, ax = plt.subplots(figsize=(,4))
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax.set_ylabel(r'$d(a,p)$')
ax.set_xlabel(r'$\frac{||a-p||}{r}$')
ax.plot(p - a, d)
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1.25)
fig.tight_layout()
fig.savefig('ARC2019/atom_density.png', bbox_inches='tight')

