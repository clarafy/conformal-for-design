# Conformal prediction under feedback covariate shift for biomolecular design
This repo contains the code accompanying the following paper:

C. Fannjiang, S. Bates, A. Angelopoulos, J. Listgarten, M. I. Jordan, Conformal prediction under feedback covariate shift for biomolecular design. 2022. *Proceedings of the National Academy of Sciences*, 119(43), e2204569119.
[publication](https://www.pnas.org/doi/10.1073/pnas.2204569119)

See `calibrate.py` for implementations of the full and split conformal prediction algorithms we describe. `assay.py` contains classes for handling the fluorescence and AAV datasets, which are stored (along with relevant saved models and results) in `fluorescence\/' and `aav\/`, respectively.

Notebooks for reproducing and plotting the results of the simulated protein design experiments are as follows:
- `notebooks/fluorescence-experiments.ipynb` shows how we ran the fluorescent protein design experiments, which uses full conformal prediction under feedback covariate shift, algorithmically optimized for ridge regression (Alg. S2 in the [SI Appendix](https://www.pnas.org/doi/10.1073/pnas.2204569119#supplementary-materials)).
- `notebooks/fluorescence-figures.ipynb` creates Figs. 3 and 4 in the main paper and Fig. S2 in the [SI Appendix](https://www.pnas.org/doi/10.1073/pnas.2204569119#supplementary-materials).
- `notebooks/aav-experiments.ipynb` shows how we ran the AAV design experiments, which uses a randomized version of split conformal prediction under covariate shift (Alg. S1 in the [SI Appendix](https://www.pnas.org/doi/10.1073/pnas.2204569119#supplementary-materials)).
- `notebooks/aav-figures.ipynb` creates Fig. 5 in the main paper.

