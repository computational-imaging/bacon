A few folks have asked how the shape spectra are visualized so here is some demo code that should allow you to reproduce those figures for the armadillo scene.

Prerequisites: 
- create a conda environment from the environment.yml file
- install imagemagick and convert utility https://imagemagick.org/script/convert.php 
- install chimerax from https://www.cgl.ucsf.edu/chimerax/ and make sure the executible is on the path as 'chimerax'

Then, run `get_shape_spectra.py` which will run the model to extract the `armadillo` scene mesh and then visualize the fourier transform
