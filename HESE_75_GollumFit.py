# Performing GollumFit on generated dataset based on the HESE-7.5 year Data Release 

import GollumFitPy as gf
import numpy as np
import os
import sys
import h5py
from collections import OrderedDict
import scipy.stats as stats

sys.path.append("Data/HESE/HESE 7.5 Data Release")
import data_loader 


np.random.seed(100) # setting a random seed for reporducibility


# Comment out different sections of the code based on what you want to run:
# 1. Generating the .fastmc file
# 2. Generating the fake data
# 3. Performing the GollumFit analysis 



######################################################################################
# GENERATING THE .FASTMC FILE THAT WE WILL BE USING AS THE MC FILES FOR OUR ANALYSIS #
######################################################################################


# ------------------------------------------------------------
#  Configure Data Paths - Set paths for cross section splines
# ------------------------------------------------------------

datapaths = gf.DataPaths()
gollumdir = "GollumFit/GollumFit"


datapaths.neutrino_cc_xs_spline_path             = gollumdir + "/resources/Splines/CrossSections/sigma_nu_CC_iso.fits"
datapaths.antineutrino_cc_xs_spline_path         = gollumdir + "/resources/Splines/CrossSections/sigma_nubar_CC_iso.fits"
datapaths.neutrino_nc_xs_spline_path             = gollumdir + "/resources/Splines/CrossSections/sigma_nu_NC_iso.fits"
datapaths.antineutrino_nc_xs_spline_path         = gollumdir + "/resources/Splines/CrossSections/sigma_nubar_NC_iso.fits"
datapaths.diff_neutrino_cc_xs_spline_path        = gollumdir + "/resources/Splines/CrossSections/dsdxdy_nu_CC_iso.fits"
datapaths.diff_antineutrino_cc_xs_spline_path    = gollumdir + "/resources/Splines/CrossSections/dsdxdy_nubar_CC_iso.fits"
datapaths.diff_neutrino_nc_xs_spline_path        = gollumdir + "/resources/Splines/CrossSections/dsdxdy_nu_NC_iso.fits"
datapaths.diff_antineutrino_nc_xs_spline_path    = gollumdir + "/resources/Splines/CrossSections/dsdxdy_nubar_CC_iso.fits"
datapaths.mc_path                                = gollumdir + "/monte_carlo/"
datapaths.domeff_spline_path                     = gollumdir + "/resources/Splines/DOMEffSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.holeice_spline_path                    = gollumdir + "/resources/Splines/HoleIceSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.attenuation_spline_path                = gollumdir + "/resources/Splines/AttenuationSplines/new_ddmnodeis"
datapaths.ice_gradient_spline_path               = gollumdir + "/resources/Splines/IceGradientsSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.atmospheric_density_spline_path        = gollumdir + "/resources/Splines/AtmosphericZenithVariationSplines/atm_density_1s.fits"
datapaths.atmospheric_kaonlosses_spline_path     = gollumdir + "/resources/Splines/AtmosphericKaonLossesSplines/kaon_loses_1s.fits"


# -------------------------------------------------------------------------------
#  Configure Flux Files - Load atmospheric, prompt, and astrophysical flux files
# -------------------------------------------------------------------------------

datapaths.conventional_nusquids_atmospheric_file = gollumdir + "/examples/fluxes/atmospheric.hdf5"
datapaths.prompt_nusquids_atmospheric_file       = gollumdir + "/examples/fluxes/prompt_atmospheric.hdf5"
datapaths.astro_nusquids_file                    = gollumdir + "/examples/fluxes/astro.hdf5"

# Hadronic and cosmic ray correction splines (necessary for flux nuisance parameters)
hadronlist = ["he_K+", "he_K-", "vhe1_pi+", "vhe1_pi-", "vhe3_K+", "vhe3_K-", 
              "vhe3_pi+", "vhe3_pi-", "vhe3_p", "vhe3_n"]
crlist = ["GSF_1", "GSF_2", "GSF_3", "GSF_4", "GSF_5", "GSF_6"]

datapaths.hadronic_spline_path   = gollumdir + "/examples/fluxes"
datapaths.cosmic_ray_spline_path = gollumdir + "/examples/fluxes"


# -------------------------------------------------------------------
#  Set Steering Parameters - Configure analysis binning and settings
# -------------------------------------------------------------------

# NOTE: Binning choices affect FastMC compression and must match analysis configuration


# Here we set the steering_params according to the configurations provided in the HESE-7.5 analysis

steering_params = gf.SteeringParams()
steering_params.minFitEnergy                    = 60000  # 60 TeV
steering_params.maxFitEnergy                    = 1e7  # 10 PeV
steering_params.logEbinEdge                     = np.log10(60000)
steering_params.logEbinWidth                    = 0.111
steering_params.minCosth                        = -1.0
steering_params.maxCosth                        = 1.0
steering_params.cosThbinEdge                    = 0.0
steering_params.cosThbinWidth                   = 0.2
steering_params.selectionStart                  = 0.99
steering_params.ice_gradient_filename           = ["Amp_0", "Amp_1", "Amp_2", "Amp_3", "Amp_4", 
                                                   "Phs_1", "Phs_2", "Phs_3", "Phs_4"]
steering_params.active_hadronic_parameters      = hadronlist
steering_params.active_cosmicray_parameters     = crlist

# Livetime for the corresponding Monte Carlo
years = 7.19
steering_params.fullLivetime                    = years * 365 * 24 * 60 * 60.
steering_params.simToLoad                       = "BDT_Split_HE"
steering_params.energyName                      = "DnnEnergy"
steering_params.model_label                     = ""  # Can be used for uniquely-labelled flux files


# ----------------------------
#  Construct and Write FastMC
# ----------------------------

gollumfit = gf.GollumFit(datapaths, steering_params)

# Here we don't use the metascaling and FastMC feature since we want all the possible 
# MC events that we get. Uncomment the next three lines of code if you want to use 
# the FastMC features


# Compression parameter: smaller values = higher compression but potential accuracy loss
# metascaling = 0.25
# gollumfit.ConstructFastMode(metascaling)


# Write to file
gollumfit.WriteCompact("Data/GollumFit_Data/HESE_75.fastmc")

print("Done generating FastMC.")



# ===========================================================================================================================================



##################################################################
# GENERATING THE TEST DATASET BASED ON THE HESE-7.5 DATA RELEASE #
##################################################################

print("Generating Test Data")


# --------------------------------------------------------------------------
#  Define Nuisance Parameters
#  Format: [vary_flag, prior_type, center, width, lower_bound, upper_bound]
# --------------------------------------------------------------------------

syst_dict     = OrderedDict({ 
    'convNorm'                  : [ True, 'Gaussian',      1.01,   0.68/2,                   0.1,                   3. ], 
    'zenithCorrection'          : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'kaonLosses'                : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'hadronicHEkp'              : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicHEkm'              : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE1pip'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE1pim'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3kp'            : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3km'            : [ True, 'Gaussian',      0.,    1.,                  -1.5,                   2. ], 
    'hadronicVHE3pip'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3pim'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3p'             : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3n'             : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'cosmicRay1'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay2'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay3'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay4'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay5'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay6'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'icegrad0'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad1'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad2'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad3'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad4'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad5'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad6'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad7'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad8'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'domEfficiency'             : [ True, 'Gaussian',    1.27, 0.123,                 1.234,                1.346 ], 
    'holeiceForward'            : [ True, 'Gaussian',     -1.,   10.,                 -5.35,                 1.85 ], 
    'astroNorm'                 : [ True, 'Gaussian',     6.37,  3.08/2,                    4.,                   8. ], 
    'astroDeltaGamma'           : [ True, 'Gaussian',      0.37,  0.4/2,                   -2.,                   2. ], 
    'astroDeltaGammaSec'        : [ True, 'Gaussian',      0.37,  0.4/2,                   -2.,                   2. ], 
    'nuxs'                      : [ True, 'Gaussian',      1.,   0.1,                 0.824,                1.176 ], 
    'nubarxs'                   : [ True, 'Gaussian',      1.,   0.1,                 0.824,                1.176 ], 
    'astroPivot'                : [ True,  'Uniform',      5.,    1./2,                    4.,                   6. ], 
    'promptNorm'                : [ True, 'Gaussian',      0.,    5.34/2,                    0.,                   3. ],
    'NeutrinoAntineutrinoRatio' : [ True, 'Gaussian',      1.,    1./2,                    0.,                   2. ],
})


# ------------------------------------------
#  set paths to relevant splines and fastMC
# ------------------------------------------

datapaths = gf.DataPaths()

gollum_dir = "GollumFit/GollumFit"

# We don't want to use the splines that came with GollumFit since they are not for
# the energy levels we are dealing with

datapaths.domeff_spline_path      = "" #gollum_dir + "/resources/Splines/DOMEffSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.holeice_spline_path     = "" #gollum_dir + "/resources/Splines/HoleIceSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.attenuation_spline_path = "" #gollum_dir + "/resources/Splines/AttenuationSplines/new_ddmnodeis"

datapaths.compact_file_path       = "Data/GollumFit_Data/HESE_75.fastmc"


# ------------------------------------
#  steering params to set the binning 
# ------------------------------------

steering_params = gf.SteeringParams()
steering_params.minFitEnergy                    = 60000
steering_params.maxFitEnergy                    = 1e7
steering_params.logEbinEdge                     = np.log10(60000)
steering_params.logEbinWidth                    = 0.111
steering_params.minCosth                        = -1.0
steering_params.maxCosth                        = 1.0
steering_params.cosThbinEdge                    = 0.0
steering_params.cosThbinWidth                   = 0.2
steering_params.selectionStart                  = float("DnnEnergy_0.99".split("_")[1])
steering_params.evalThreads                     = 1


# --------------------------
#  declare gollumfit object
# --------------------------

gollumfit = gf.GollumFit(datapaths,steering_params)

fitparams = gf.FitParameters()

# set the nuisance parameter values
for sname in syst_dict.keys():
    if sname: 
        exec('fitparams.'+sname+' = syst_dict[\"'+sname+'\"][2]')

null_dist = gollumfit.GetExpectationEvents(fitparams)


# ----------------------- 
#  saving to a .npz file
# -----------------------

print('saving to npz file.')
realization = "Data/GollumFit_Data/modified_fake_data.npz"
np.savez(realization, realization=null_dist)


print("Done. Data saved to "+realization)



# ===========================================================================================================================================



####################################################
# PERFORMING THE GOLLUMFIT LIKELIHOOD MINIMIZATION #
####################################################


# --------------------------------------------------------------------------
#  Define Nuisance Parameters (All set to vary in fit with False flag)
#  Format: [vary_flag, prior_type, center, width, lower_bound, upper_bound]
# --------------------------------------------------------------------------


syst_dict     ={ 
    'convNorm'                  : [ False, 'Gaussian',      1.01,   0.68/2,                   0.1,                   3. ], 
    'zenithCorrection'          : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'kaonLosses'                : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'hadronicHEkp'              : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicHEkm'              : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE1pip'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE1pim'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3kp'            : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3km'            : [ True, 'Gaussian',      0.,    1.,                  -1.5,                   2. ], 
    'hadronicVHE3pip'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3pim'           : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3p'             : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'hadronicVHE3n'             : [ True, 'Gaussian',      0.,    1.,                   -2.,                   2. ], 
    'cosmicRay1'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay2'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay3'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay4'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay5'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'cosmicRay6'                : [ True, 'Gaussian',      0.,    1.,                   -4.,                   4. ], 
    'icegrad0'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad1'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad2'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad3'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad4'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad5'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad6'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad7'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'icegrad8'                  : [ True, 'Gaussian',      0.,    1.,                   -3.,                   3. ], 
    'domEfficiency'             : [ True, 'Gaussian',    1.27, 0.123,                 1.234,                1.346 ], 
    'holeiceForward'            : [ True, 'Gaussian',     -1.,   10.,                 -5.35,                 1.85 ], 
    'astroNorm'                 : [ False, 'Gaussian',      6.37,  3.08/2,                    1.,                   11. ], 
    'astroDeltaGamma'           : [ False, 'Gaussian',      0.37,  0.4/2,                   -2.,                   2. ], 
    'astroDeltaGammaSec'        : [ False, 'Gaussian',      0.37,  0.4/2,                   -2.,                   2. ], 
    'nuxs'                      : [ False, 'Gaussian',      1.,   0.1,                 0.824,                1.176 ], 
    'nubarxs'                   : [ False, 'Gaussian',      1.,   0.1,                 0.824,                1.176 ], 
    'astroPivot'                : [ False,  'Uniform',      5.,    1./2,                    4.,                   6. ], 
    'promptNorm'                : [ False, 'Gaussian',      0.,    5.34/2,                    0.,                  3. ],
    'NeutrinoAntineutrinoRatio' : [ False, 'Gaussian',      1.,    1./2,                    0.,                   2. ],
}



# ----------------------------------------------------------------------------
#  Define Random Sampling Function
#  Helper function to sample random parameter values from prior distributions
# ----------------------------------------------------------------------------

# NOTE: Modified throw() so that it outputs the mean value of the parameter if the parameter is non-varying
# and randomly initializes the varying parameters 

def throw(syst):
    """Sample random value from prior distribution."""

    if syst[0] == False:
        if syst[1] == 'Gaussian':
            # Truncated normal distribution
            val = stats.truncnorm(
                (syst[4] - syst[2]) / syst[3],  # Lower bound (standardized)
                (syst[5] - syst[2]) / syst[3],  # Upper bound (standardized)
                syst[2],  # Mean
                syst[3]   # Std dev
            ).rvs(1)
            return val[0]
        else:
            # Uniform distribution
            return np.random.uniform(syst[4], syst[5])
    else:
        return syst[2]


# --------------------------------------------
#  Initialize Fit Parameter Objects
#  Create objects to manage fit configuration
# --------------------------------------------

fitparams_flag  = gf.FitParametersFlag()  # Which parameters to vary
fitparams_bound = gf.FitParametersBound()  # Parameter bounds
priors          = gf.Priors()              # Prior distributions
seed_fitparams  = gf.FitParameters()       # Initial values


# --------------------------------------------------------------------
#  Set Priors and Random Initial Values
#  Configure priors and randomly initialize starting parameter values
# --------------------------------------------------------------------

print('Initializing with the following randomly-seeded nuisance params:')

for sname in syst_dict.keys():
    # Set flags and bounds
    exec(f'fitparams_flag.{sname} = syst_dict["{sname}"][0]')
    exec(f'fitparams_bound.{sname}Min = syst_dict["{sname}"][4]')
    exec(f'fitparams_bound.{sname}Max = syst_dict["{sname}"][5]')
    
    # Set priors
    if syst_dict[sname][1] == 'Gaussian':
        exec(f'priors.{sname}Center = syst_dict["{sname}"][2]')
        exec(f'priors.{sname}Width  = syst_dict["{sname}"][3]')
    else:
        exec(f'priors.{sname}Min = syst_dict["{sname}"][4]')
        exec(f'priors.{sname}Max = syst_dict["{sname}"][5]')
    
    # Randomly initialize
    thrown_val = throw(syst_dict[sname])
    exec(f'seed_fitparams.{sname} = thrown_val')
    print(f'{sname}: {thrown_val}')


# -----------------------------------------------------------------
#  Set Correlations and Paths
#  Load correlation matrices for ice gradients and flux parameters
# -----------------------------------------------------------------

gollumdir = "GollumFit/GollumFit"

# Set correlations (required for fitting/minimization, not for likelihood evaluation)

iceg_corr = np.load(gollumdir + '/resources/correlation_matrices/icegrad_correlations.npy')
flux_corr = np.load(gollumdir + '/resources/correlation_matrices/flux_correlations_new_ddmnodeis.npy')

for idx, val in np.ndenumerate(iceg_corr):
    priors.SetIceGradientsCorr(idx[0], idx[1], val)
for idx, val in np.ndenumerate(flux_corr):
    priors.SetFluxCorr(idx[0], idx[1], val)

datapaths = gf.DataPaths()

# We don't want to use the splines that came with GollumFit since they are not for
# the energy levels we are dealing with

datapaths.domeff_spline_path      = "" #gollumdir + "/resources/Splines/DOMEffSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.holeice_spline_path     = "" #gollumdir + "/resources/Splines/HoleIceSplines/new_ddmnodeis/BDT/DnnEnergy_0.99"
datapaths.attenuation_spline_path = "" #gollumdir + "/resources/Splines/AttenuationSplines/new_ddmnodeis"

datapaths.compact_file_path       = 'Data/GollumFit_Data/HESE_75.fastmc'


# ------------------------------------------------------------------
#  Configure Steering Parameters
#  Set binning and convergence criteria (must match FastMC binning)
# ------------------------------------------------------------------

steering_params = gf.SteeringParams()
steering_params.minFitEnergy                    = 60000
steering_params.maxFitEnergy                    = 1e7
steering_params.logEbinEdge                     = np.log10(60000)
steering_params.logEbinWidth                    = 0.111
steering_params.minCosth                        = -1.0
steering_params.maxCosth                        = 1.0
steering_params.cosThbinEdge                    = 0.0
steering_params.cosThbinWidth                   = 0.2
steering_params.selectionStart                  = float("DnnEnergy_0.99".split("_")[1])
steering_params.evalThreads                     = 1

# Convergence criteria (tight tolerances for accurate minimization)
steering_params.change_tol     = 1.e-5 #1.e-20
steering_params.grad_tol       = 1.e-5 #1.e-20
steering_params.uncertaintyModSigmaOverMu = 0.0


# ----------------------------------------------
#  Load Data and Configure Fit
#  Create GollumFit object and load pseudo-data
# ----------------------------------------------

gollumfit = gf.GollumFit(datapaths, steering_params)

realization  = "Data/GollumFit_Data/modified_fake_data.npz"
total_data = gollumfit.SetData(np.load(realization)["realization"])


# ---------------------------------------------------------------------------
#  feed the flags, bounds, priors, on the nuisance parameters into gollumfit
# ---------------------------------------------------------------------------

gollumfit.SetFitParametersFlag(fitparams_flag)
gollumfit.SetFitParametersBound(fitparams_bound)
gollumfit.SetFitParametersPriors(priors)
gollumfit.SetFitParametersSeed([seed_fitparams])
gollumfit.ConstructLikelihoodProblem()


# ---------------------------------
#  perform the global minimization
# ---------------------------------

print("Starting global minimization...")
min_llh = gollumfit.MinLLH()

# ------------------------------------------------------------------------------------------------
#  results: print the best fit nuisance parameters, likelihood, and the number of LLH evaluations
# ------------------------------------------------------------------------------------------------

systematics = ""
for sname in syst_dict.keys() :
    exec('print(\"'+sname+'\",min_llh.params.'+sname+')')
    exec('systematics += str(min_llh.params.'+sname+')+\" \"')


