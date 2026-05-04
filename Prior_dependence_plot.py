import numpy as np
import matplotlib.pyplot as plt
import GollumFitPy as gf
import scipy.stats as stats
import os
from collections import OrderedDict

#####################################################################################
# Configuration for Prior Dependence Test
#####################################################################################
PARAM_TO_TEST = 'astroDeltaGamma'  # Change this to whatever parameter you want to test
TRUE_VALUE = 0.37
PRIOR_MEANS = np.linspace(max(TRUE_VALUE - 1.5, 0), TRUE_VALUE + 1.5, 15)  # The prior mean values to scan over
RESULTS_FILE = 'Data/GollumFit_Data/prior_dependence/' + PARAM_TO_TEST +'_priors.npz'
#####################################################################################

# Ensure output directory exists
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

syst_dict ={ 
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

# Ensure the parameter we're testing is varied (False)
syst_dict[PARAM_TO_TEST][0] = False

#####################################################################################
# Set paths to relevant splines and fastMC
#####################################################################################
datapaths = gf.DataPaths()
gollumdir = "GollumFit/GollumFit"
datapaths.domeff_spline_path      = "" 
datapaths.holeice_spline_path     = "" 
datapaths.attenuation_spline_path = "" 
datapaths.compact_file_path       = "Data/GollumFit_Data/HESE_75.fastmc"

#####################################################################################
# Steering params to set the binning 
#####################################################################################
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
steering_params.change_tol                      = 1.e-5
steering_params.grad_tol                        = 1.e-5
steering_params.uncertaintyModSigmaOverMu       = 0.0

#####################################################################################
# Initialize GollumFit
#####################################################################################
gollumfit = gf.GollumFit(datapaths, steering_params)

realization  = "Data/GollumFit_Data/modified_fake_data.npz"
total_data = gollumfit.SetData(np.load(realization)["realization"])

# Load correlation matrices once
iceg_corr = np.load(os.path.join(gollumdir, 'resources/correlation_matrices/icegrad_correlations.npy'))
flux_corr = np.load(os.path.join(gollumdir, 'resources/correlation_matrices/flux_correlations_new_ddmnodeis.npy'))


fit_results = []
seed_fitparams  = gf.FitParameters()

for sname in syst_dict.keys():
    # Set the seed for non-tested params strictly to their true/nominal center to isolate prior dependence 
    exec(f'seed_fitparams.{sname} = syst_dict["{sname}"][2]')

# Loop over the prior means
for i, prior_mean in enumerate(PRIOR_MEANS):
    print(f"[{i+1}/{len(PRIOR_MEANS)}] Running fit with {PARAM_TO_TEST} Prior Mean = {prior_mean}")
    
    priors          = gf.Priors()
    fitparams_flag  = gf.FitParametersFlag()
    fitparams_bound = gf.FitParametersBound()

    # Reconstruct the fit arrays with the modified prior mean
    for sname in syst_dict.keys():
        exec(f'fitparams_flag.{sname} = syst_dict["{sname}"][0]')
        exec(f'fitparams_bound.{sname}Min = syst_dict["{sname}"][4]')
        exec(f'fitparams_bound.{sname}Max = syst_dict["{sname}"][5]')
        
        # Default initialization for priors
        if syst_dict[sname][1] == 'Gaussian':
            exec(f'priors.{sname}Center = syst_dict["{sname}"][2]')
            exec(f'priors.{sname}Width  = syst_dict["{sname}"][3]')
        else:
            exec(f'priors.{sname}Min = syst_dict["{sname}"][4]')
            exec(f'priors.{sname}Max = syst_dict["{sname}"][5]')
            
    # Override the prior mean for the parameter we are testing
    if syst_dict[PARAM_TO_TEST][1] == 'Gaussian':
        exec(f'priors.{PARAM_TO_TEST}Center = prior_mean')
    else:
        print(f"Warning: {PARAM_TO_TEST} doesn't have a Gaussian prior, changing bounds instead?! Fix not implemented for uniform prior shifts.")
    
    # We will also initialize the seed at the prior mean for the tested parameter to match normal throwing behavior
    exec(f'seed_fitparams.{PARAM_TO_TEST} = prior_mean')
    
    # Apply correlation matrices
    for idx, val in np.ndenumerate(iceg_corr):
        priors.SetIceGradientsCorr(idx[0], idx[1], val)
    for idx, val in np.ndenumerate(flux_corr):
        priors.SetFluxCorr(idx[0], idx[1], val)
        
    gollumfit.SetFitParametersFlag(fitparams_flag)
    gollumfit.SetFitParametersBound(fitparams_bound)
    gollumfit.SetFitParametersPriors(priors)
    gollumfit.SetFitParametersSeed([seed_fitparams])
    gollumfit.ConstructLikelihoodProblem()
    
    # Run Minimization
    min_llh = gollumfit.MinLLH()
    
    # Extract the fitted value
    best_fit_val = None
    exec(f'best_fit_val = min_llh.params.{PARAM_TO_TEST}')
    
    print(f" -> Best Fit Value for {PARAM_TO_TEST} = {best_fit_val:.4f}\n")
    fit_results.append(best_fit_val)
    

# Save the dataset
np.savez(RESULTS_FILE, 
         prior_means=PRIOR_MEANS, 
         fit_results=fit_results, 
         param=PARAM_TO_TEST, 
         true_value=TRUE_VALUE)

print(f"All fits completed! Results saved to {RESULTS_FILE}")




inp =  input("Proceed with plotting the prior dependence graph? (Y/n): ")
if inp != "" and inp != "Y" and inp != "y":
    exit() 



# Plotting the prior dependence

#####################################################################################
# Load Data
#####################################################################################
data = np.load(RESULTS_FILE)
prior_means = data['prior_means']
fit_values = data['fit_results']
PARAM_TO_TEST = str(data['param'])
TRUE_VALUE = float(data['true_value'])
PLOT_OUT = 'Data/GollumFit_Data/prior_dependence/' + PARAM_TO_TEST + '_prior_plot.png'

#####################################################################################
# Generate Plot
#####################################################################################
plt.figure(figsize=(8, 6))

# Plot the best fit vs prior
plt.plot(prior_means, fit_values, 'o-', color='blue', label='Best Fit Values', linewidth=2, markersize=8)

# Plot the True value line
plt.axhline(TRUE_VALUE, color='red', linestyle='--', label=f'True Value ({TRUE_VALUE})', linewidth=2)

# Plot y = x line (where Fit Value completely tracks Prior Mean)
plt.plot(prior_means, prior_means, 'k:', label='y = x (Complete Prior Dependence)', linewidth=2)

# Formatting the plot
plt.xlabel(f'Prior Mean for {PARAM_TO_TEST}', fontsize=12)
plt.ylabel(f'Fitted Value for {PARAM_TO_TEST}', fontsize=12)
plt.title(f'Prior Dependence of {PARAM_TO_TEST}', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5)
plt.tight_layout()

# Save the plot
os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)
plt.savefig(PLOT_OUT, dpi=300)
print(f"Plot successfully saved to: {PLOT_OUT}")

plt.show() # Uncomment if you want the interactive plot window to pop up

