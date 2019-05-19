r"""
============================================
EX3: LOL from Molecule and user-defined cube
============================================

Compute LOL and visualize it for formamide, using the inverse hyperbolic transformation.

"""

from chemtools import LOL, UniformGrid, Molecule

# 1. Build Molecule, UnifromGrid and LOL model

mol = Molecule.from_file('chonh2.fchk')
cub = UniformGrid.from_molecule(mol, spacing=0.1, extension=2.0)
lol = LOL.from_molecule(mol, grid=cub, trans='inverse_hyperbolic', trans_k=1, trans_a=1)

# 2. Generate cube file(s) and script for visualizing LOL
#    Files generated are chonh2-lol.cube & chonh2.vmd
#    To visualize the iso-surface, use command: $ vmd -e chonh2.vmd

lol.generate_scripts('chonh2', isosurf=0.55)


# DISCARD BELOW:
# the code below is for displaying the LOL image on the website, you should remove it
# when running the script on your machine.
from tools.rug import plot_existing_image

plot_existing_image('lol055_chonh2_hyperbolic.jpg')
