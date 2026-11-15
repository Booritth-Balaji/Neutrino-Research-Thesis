# High-Energy Neutrinos: A Source and Flux Analysis

**Student**: T.Y. Booritth Balaji

**Supervisor**: Prof. Ranjan Laha (Department of Physics, IISc)

**Degree**: Bachelor of Science (Research)

**University**: Indian Institute of Science (IISc), Bengaluru, India

<!-- Bachelor of Science (Research) thesis, Indian Institute of Science
**T.Y. Booritth Balaji**, supervised by **Prof. Ranjan Laha** (Department of Physics, IISc) -->

### 📄 [Read the thesis (PDF)](Bachelor_Thesis.pdf)

---

## About

This repository contains the code, data and notes for my bachelor's thesis on high-energy astrophysical neutrinos. The work has two parts:

1. **PeV neutrino analysis.** I model the atmospheric conventional, atmospheric prompt and astrophysical neutrino fluxes, estimate their event-rate spectra in IceCube, and work out how likely each one is to be the source of the first two PeV cascade events IceCube detected.
2. **HESE events and GollumFit.** I study IceCube's High-Energy Starting Events (HESE) and use GollumFit's likelihood framework to test whether a mock dataset resembling HESE-7.5 can be fit reliably.

## Which files to use

| File | What it's for |
|---|---|
| `PeV_events_plots.ipynb` | Part 1: flux models, event-rate spectra and the PeV event probabilities |
| `Starting_events_plots.ipynb` | Sky maps of the HESE 7.5 and HESE 12 events |
| `GollumFit_Examples.ipynb` | Walkthrough of the GollumFit workflow: FastMC generation, fake data, likelihood minimisation |
| `HESE_75_GollumFit.py` | Main Part 2 script: generates the `.fastmc` file and the mock HESE-7.5 data, then runs the fit (comment sections in or out to choose what runs) |
| `Prior_dependence_plot.py` | Tests how the fit depends on the prior for a chosen parameter |
| `GollumFit_plots.ipynb` | Plots of the fit results and the data histograms |
| `Neutrino_sky_plots.ipynb` | IceCube 2008–2018 sky map with galaxy-merger catalogues overlaid (exploratory) |
| `Rough.ipynb` | Rough calculations and scratch work |

Data lives in `Data/`, and the main reference papers are in `Papers/`.

## Running the code

- Run the notebooks and scripts **from the repository root**. All file paths in them are relative to it.
- Python packages used: `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`.
- The `.npz` data files are stored with Git LFS, so run `git lfs pull` after cloning.
- The GollumFit parts (`HESE_75_GollumFit.py`, `Prior_dependence_plot.py`, `GollumFit_*.ipynb`, `Rough.ipynb`) need the `GollumFitPy` module. `GollumFit/local/` holds the build I used (Linux, Python 3.10). The GollumFit source itself (`GollumFit/GollumFit`) isn't included in this repository and has to be obtained separately.

## Data sources

- **IceCube HESE 7.5-year public data release:** R. Abbasi et al., *Phys. Rev. D* 104, 022002 (2021). It's included in `Data/HESE/HESE 7.5 Data Release/` with its original README and LICENSE.
- **IceCube HESE 12-year events:** T. Yuan & D. Chirkin, arXiv:2307.13878.
- **IceCube 2008–2018 point-source event data:** `Data/IceCube_2008-2018_double_counting_corrected/`.
- **Fluxes, cross-sections and effective areas** for the PeV analysis, digitised from the papers cited in the thesis, mainly R. Laha et al., *Phys. Rev. D* 88, 043009 (2013) and M. Honda et al. (2007).
- **Galaxy merger catalogues:** from the VizieR catalogues named in `Data/galaxy_mergers_cats/`.
- **GollumFit, LeptonWeighter and nuSQuIDS:** IceCube Collaboration software.
