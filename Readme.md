### Intro

This repository contains the code for the analytic model in https://arxiv.org/abs/2403.00912. 

Required packages: python, numpy, scipy, astropy, matplotlib for plotting in notebook.


### Description

See `example.ipynb` for example usage and how to reproduce the figures in the paper.

`bathtub_model.py` contains the bathtub model described in sec 5.4 and originally introduced in https://arxiv.org/abs/1402.2283.  

`halo_model.py` contains the model for halo profiles described in sec 5.1. 

`filament_model.py` contains the model for filament profiles described in sec 5.2, appendix B and associated `filament_model.pdf`. 

`filament_model.pdf` elaborates on the self-similar filament profile described in appendix B. Keshav Raghavan is heavily involved in the development of this model.

### Citation

Please cite https://ui.adsabs.harvard.edu/abs/2024arXiv240300912A/exportcitation if you use this code.