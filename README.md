# axionJWST_2024 (Summer project 2024)

This repos is to analyse the data from MIRI JWST and perform axion DM spectrocopy.

Codes:
1. MRS_main.py
    Main for performing spectroscopic data analysis
2. MRS_func.py
    Functions for performing spectroscopic data analysis. Currently using multiprocessing pool for fittings of different energy windows. Computational efficiency is unexpectedly low.
3. MRS_func_nopool.py
    Remove the multiprocessing pool function from the standard function file. Efficiency is increased.
4. stackSpec.ipynb
    To read a directory containing fits files of different observations downloaded from MAST achive, and to stack all the read data files. Current problems: how to safely obtain as many data as possible and how to increase SNR.
5. plotResult.ipynb
    Visualize the result file (in .npz format) produced by MRS_main.py


Data files:
Processed data files are stored in analysis_data (.npz files). Raw files have to be obtained from MAST directly.
