"""
ngsf_comparison.py
==================
Runs the SAME N_TRIALS simulated spectra through both:
  1. Your sncosmo-based Superfit classifier
  2. NGSF (Next Generation SuperFit)

Records per trial:
  true_template, true_type, true_phase, snr
  sf_predicted_type,   sf_predicted_phase,   sf_chi2
  ngsf_predicted_type, ngsf_predicted_phase, ngsf_chi2

Produces side-by-side confusion matrices for direct comparison.

USAGE:
  In Terminal (keeps Mac awake even with lid closed):
      caffeinate -i python ngsf_comparison.py

  Resumable — if interrupted, delete comparison_progress.csv to start fresh,
  or leave it to resume from where it stopped.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import sncosmo
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

# ============================================================
# !! EDIT THIS PATH !!
# ============================================================
PARAMS_PATH = "/Users/neelviswanathan/PycharmProjects/NGSF Overall copy/parameters.json"

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_DIR  = os.path.expanduser("~/superfit_comparison_resuls")
os.makedirs(RESULTS_DIR, exist_ok=True)

Z            = 0.05
N_TRIALS     = 500
SNR_RANGE    = (10, 90)
PHASE_RANGE  = (-15, 40)
WAVE_RANGE   = (4000, 9000)
N_WAVE       = 2000
BIN_WIDTH_AA = 10.0
SEED         = 30

# ============================================================
# MODEL DICTIONARY  (IIP+IIL merged into II)
# ============================================================

model_dict = {
    'v19-2014g': 'II', 's11-2005lc': 'II', 'snana-2004ib': 'II', 'v19-1987a-corr': 'II-pec',
    'snana-2006ix': 'IIn', 'snana-2007ky': 'II', 'v19-2013df-corr': 'IIb', 'v19-2004aw': 'Ic',
    'v19-2006ep-corr': 'Ib', 'v19-2004et': 'II', 'v19-2012au-corr': 'II', 'whalen-z25b': 'PopIII-II',
    'v19-2013ge-corr': 'Ic', 'v19-2012au': 'II', 'snana-2006gq': 'Ic', 'snana-2007nv': 'II',
    'v19-1999em': 'II', 'v19-2012aw': 'II', 'snemo15': 'Ia', 'v19-2004gq': 'Ic',
    'nugent-sn1bc': 'Ib/Ic', 'v19-2011fu': 'IIb', 'v19-2016bkv': 'II', 'v19-1993j': 'IIb',
    'v19-iptf13bvn-corr': 'Ib', 'v19-2004aw-corr': 'Ic', 'v19-2006aj': 'Ic-BL', 'v19-2011ei': 'Ib/Ic',
    's11-2005hm': 'Ib', 'v19-2005bf': 'Ib', 'v19-2004fe-corr': 'Ic', 'salt3': 'Ia',
    'v19-2011ht': 'IIn', 'snana-04d1la': 'II', 'snana-2005gi': 'II', 'v19-2011hs': 'IIb',
    'snana-sdss014475': 'II', 'v19-2005hg-corr': 'Ib', 'snana-2007kw': 'II', 'hsiao-subsampled': 'Ia',
    'v19-1999dn-corr': 'Ib/Ic', 'snana-2006ns': 'II', 'mlcs2k2': 'Ia', 'v19-2011hs-corr': 'IIb',
    'v19-2013ab-corr': 'II', 'v19-2009bb-corr': 'Ic-BL', 'v19-2013ab': 'II', 'v19-2013fs': 'II',
    'nugent-sn1a': 'Ia', 'salt2-extended': 'Ia', 'v19-1993j-corr': 'IIb', 'v19-2009ip-corr': 'IIn',
    'snana-2007og': 'IIn', 'v19-2004gv': 'Ib', 'v19-2007od-corr': 'II', 'v19-2011ei-corr': 'Ib/Ic',
    'v19-2011fu-corr': 'IIb', 'v19-asassn15oz-corr': 'II', 'snana-2006jo': 'Ib', 'snana-2007lj': 'IIn',
    'snana-2007iz': 'II', 'nugent-sn2l': 'II', 'snana-2007lx': 'II', 'v19-2007y': 'Ib',
    'v19-2004gt': 'Ib/Ic', 'v19-asassn14jb': 'II', 's11-2005gi': 'II', 'v19-2009dd': 'II',
    'snana-2004gq': 'Ic', 's11-2006fo': 'Ic', 'snemo7': 'Ia', 'v19-2012ap': 'Ic-BL',
    'v19-2009iz-corr': 'Ib', 'v19-2008d-corr': 'Ib', 'snana-sdss004012': 'Ic', 'v19-2009dd-corr': 'II',
    'snana-2007nw': 'II', 'v19-2006aa': 'II', 'v19-2013fs-corr': 'II', 'v19-2016gkg-corr': 'IIb',
    'snana-2005hm': 'Ib', 'snana-2004hx': 'II', 'v19-2008ax-corr': 'IIb', 'nugent-hyper': 'Ic-BL',
    'v19-2002ap': 'Ic-BL', 'v19-2007gr': 'Ic', 'v19-2008aq-corr': 'II', 'v19-2013ge': 'Ic',
    'snana-2006kv': 'II', 'v19-2016x': 'II', 'v19-2011bm': 'Ic', 'v19-2014g-corr': 'II',
    'salt3-nir': 'Ia', 'snana-04d4jv': 'II', 'v19-2007uy': 'Ib', 'v19-2007ru': 'II',
    'v19-2002ap-corr': 'Ic-BL', 'snana-2007nr': 'II', 'v19-2011ht-corr': 'IIn', 'snana-2007lb': 'II',
    'snana-2007y': 'Ib', 'v19-2004gt-corr': 'Ib/Ic', 'v19-2011dh-corr': 'IIb', 's11-2006jo': 'Ib',
    'v19-2004et-corr': 'II', 'v19-2007pk': 'II', 'nugent-sn91bg': 'Ia-91bg', 'v19-2009jf-corr': 'Ib/Ic',
    'v19-2009bw-corr': 'II', 'v19-1987a': 'II-pec', 'snana-2007ny': 'II', 'v19-2004fe': 'Ic',
    'v19-2009kr-corr': 'II', 'v19-2007uy-corr': 'Ib', 'v19-2009ib': 'II', 'snana-2007ld': 'II',
    'whalen-z40b': 'PopIII-II', 'v19-2008ax': 'IIb', 'v19-2007pk-corr': 'II', 'v19-2008bj': 'II',
    'v19-2006t-corr': 'II', 'v19-2009iz': 'Ib', 'hsiao': 'Ia', 'v19-2016x-corr': 'II',
    'whalen-z15b': 'PopIII-II', 'whalen-z25d': 'PopIII-II', 'v19-1999em-corr': 'II', 'v19-1999dn': 'Ib/Ic',
    'whalen-z40g': 'PopIII-II', 'v19-2013by-corr': 'II', 'snana-2006ez': 'II', 'nugent-sn1superc': 'Ia-SC',
    'v19-2013ej-corr': 'II', 'snana-2007pg': 'II', 'v19-2009ip': 'IIn', 's11-2005hl': 'Ib',
    'whalen-z15d': 'PopIII-II', 'v19-2009bw': 'II', 'v19-2013df': 'IIb', 'v19-2008bj-corr': 'II',
    'snana-2007md': 'II', 'whalen-z25g': 'PopIII-II', 'nugent-sn91t': 'Ia-91t', 'v19-2004gq-corr': 'Ic',
    'snana-2006lc': 'II', 'v19-2008fq': 'II', 'v19-2009jf': 'Ib/Ic', 'v19-iptf13bvn': 'Ib',
    'v19-2008d': 'Ib', 'v19-2013am-corr': 'II', 'snana-2006ep': 'Ib', 'salt2': 'Ia',
    'v19-2010al': 'II', 'v19-2016bkv-corr': 'II', 'v19-1994i': 'Ic', 'v19-2009n': 'II',
    'v19-2009bb': 'Ic-BL', 'v19-2008in-corr': 'II', 'v19-2007od': 'II', 'v19-2008bo': 'II',
    'v19-2010al-corr': 'II', 'v19-2011dh': 'IIb', 'v19-2016gkg': 'IIb', 'snana-2004gv': 'Ib',
    'v19-2012a': 'II', 'v19-2008fq-corr': 'II', 'snana-2007lz': 'II', 'v19-2005hg': 'Ib',
    'v19-2006aa-corr': 'II', 'v19-2013ej': 'II', 'v19-2007y-corr': 'Ib', 's11-2006jl': 'IIn',
    'nugent-sn2n': 'IIn', 'snana-2006iw': 'IIn', 'snana-2007ll': 'II', 'v19-2007ru-corr': 'II',
    'v19-2009n-corr': 'II', 'snemo2': 'Ia', 's11-2004hx': 'II', 'v19-2005bf-corr': 'Ib',
    'v19-2013by': 'II', 'v19-1998bw-corr': 'Ic-BL', 'v19-2006t': 'II', 'v19-2006aj-corr': 'Ic-BL',
    'v19-2006ep': 'Ib', 'v19-1998bw': 'Ic-BL', 'v19-2008in': 'II', 'v19-2008bo-corr': 'II',
    'v19-2012aw-corr': 'II', 'v19-2013am': 'II', 'whalen-z15g': 'PopIII-II', 'v19-2011bm-corr': 'Ic',
    'v19-1994i-corr': 'Ic', 'v19-2009kr': 'II', 'v19-2008aq': 'II', 'snana-2006kn': 'IIn',
    'v19-2012ap-corr': 'Ic-BL', 'salt2-extended-h17': 'Ia', 'v19-2004gv-corr': 'Ib', 'sugar': 'Ia',
    'v19-2012a-corr': 'II', 'v19-2009ib-corr': 'II', 'snana-2006jl': 'IIn', 'nugent-sn2p': 'II',
    'salt2-h17': 'Ia', 'snana-2007ms': 'II', 'snana-2004fe': 'Ic', 'v19-asassn15oz': 'II',
    'v19-2007gr-corr': 'Ic', 'snana-2006fo': 'Ic', 'snf-2011fe': 'Ia', 'v19-asassn14jb-corr': 'II',
    'snana-2007nc': 'II',
}

SALT_SOURCES     = {'salt2','salt2-extended','salt2-h17','salt2-extended-h17','salt3','salt3-nir'}
BROKEN_SOURCES   = {'mlcs2k2'}
CLASSIFY_EXCLUDE = BROKEN_SOURCES | {
    'nugent-sn1a','nugent-sn1bc','nugent-sn2p','nugent-sn2l',
    'nugent-sn2n','nugent-hyper','nugent-sn91bg','nugent-sn91t','nugent-sn1superc',
}
T0_STARTS = [-12.0, 0.0, 20.0]

# ============================================================
# SOFT-MATCH CORRECTNESS
# ============================================================

def is_correct(true_type, predicted_type):
    if true_type == predicted_type:
        return True
    if true_type == 'Ib/Ic' and predicted_type in ('Ib', 'Ic'):
        return True
    if true_type in ('Ia-91bg', 'Ia-91t', 'Ia-SC') and predicted_type == 'Ia':
        return True
    return False

# ============================================================
# NGSF TYPE PARSING
# Confirmed: SN column = "TypeFolder/SNname/template phase-band : X.XB"
# ============================================================

NGSF_TYPE_MAP = {
    'Ia-norm': 'Ia',        'Ia 91T-like': 'Ia-91t',  'Ia 91bg-like': 'Ia-91bg',
    'Ia 99aa-like': 'Ia',   'Ia-02cx like': 'Ia',      'Ia 02es-like': 'Ia',
    'Ia-pec': 'Ia',         'Ia-rapid': 'Ia',           'Ia-CSM': 'Ia',
    'Ia-CSM-(ambigious)': 'Ia', 'super_chandra': 'Ia-SC', 'Ca-Ia': 'Ia',
    'II': 'II',             'IIn': 'IIn',               'IIb': 'IIb',
    'II-flash': 'II',       'IIb-flash': 'IIb',         'Ib': 'Ib',
    'Ibn': 'Ib',            'Ca-Ib': 'Ib',              'Ic': 'Ic',
    'Ic-BL': 'Ic-BL',      'Ic-pec': 'Ic',             'SLSN-I': 'Ic-BL',
    'SLSN-II': 'II',        'SLSN-IIn': 'IIn',          'SLSN-IIb': 'IIb',
    'SLSN-Ib': 'Ib',        'FBOT': 'Ic-BL',
}

def parse_ngsf_sn_label(sn_field):
    if not sn_field or str(sn_field).strip() == '':
        return 'FAILED'
    top = str(sn_field).strip().split('/')[0].strip()
    if top in NGSF_TYPE_MAP:
        return NGSF_TYPE_MAP[top]
    for key, val in NGSF_TYPE_MAP.items():
        if key.lower() in top.lower():
            return val
    return 'Unknown'

# ============================================================
# SUPERFIT INTERNALS
# ============================================================

def _amp_param(model_name, model_obj=None):
    if model_name in SALT_SOURCES:
        return 'x0'
    if model_obj is not None:
        pnames = list(model_obj.param_names)
        if 'amplitude' in pnames:
            return 'amplitude'
        candidates = [p for p in pnames if p not in ('z', 't0')]
        if candidates:
            return candidates[0]
    return 'amplitude'

def _extra_params(model_name, pnames):
    extra = {}
    if model_name in SALT_SOURCES:
        if 'x1' in pnames: extra['x1'] = 0.0
        if 'c'  in pnames: extra['c']  = 0.0
    for p in pnames:
        if p not in ('z', 't0') and p not in extra:
            extra[p] = 0.0
    return extra

class Spectrum:
    def __init__(self, wavelength, flux, redshift=0.05):
        self.wavelength  = np.array(wavelength, dtype=float)
        flux             = np.array(flux, dtype=float)
        pos              = flux[flux > np.percentile(flux, 10)]
        self.norm_factor = float(np.median(pos)) if len(pos) > 0 else 1.0
        self.flux        = flux / self.norm_factor
        self.redshift    = redshift
        if len(self.flux) > 4:
            diffs = np.diff(self.flux)
            mad   = np.median(np.abs(diffs - np.median(diffs)))
            sigma = max(mad * 1.4826 / np.sqrt(2), 1e-4)
        else:
            sigma = 0.1
        self.flux_error = np.full_like(self.flux, sigma)

def chi2_objective(params, model, wavelength, flux, flux_error, model_name):
    amp, t0, gal_norm, gal_index = params
    try:
        ap = _amp_param(model_name, model)
        model.set(**{ap: amp, 't0': t0})
        m_flux = model.flux(0, wavelength)
        g_flux = gal_norm * (wavelength / 5000.0) ** gal_index
        chi2   = np.sum(((flux - (m_flux + g_flux)) / flux_error) ** 2)
        return chi2 / max(len(wavelength) - 4, 1)
    except Exception:
        return 1e15

def optimize_fit(spec, model_name, z):
    try:
        model  = sncosmo.Model(source=model_name)
        pnames = list(model.param_names)
        model.set(z=z)
        ap   = _amp_param(model_name, model)
        init = _extra_params(model_name, pnames)
        init.pop(ap, None)
        try: model.set(**init)
        except Exception: pass
        best_res, best_chi2 = None, np.inf
        for t0_start in T0_STARTS:
            try:
                res = minimize(chi2_objective,
                               x0=[1.0, t0_start, 0.2, -1.0],
                               bounds=[(1e-6,100.0),(-30.0,50.0),(0.0,1.0),(-3.0,3.0)],
                               args=(model, spec.wavelength, spec.flux,
                                     spec.flux_error, model_name),
                               method='L-BFGS-B',
                               options={'maxiter':300,'ftol':1e-9})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best_res  = res
            except Exception:
                continue
        if best_res is None:
            return None
        return {
            'predicted_template': model_name,
            'predicted_type':     model_dict.get(model_name, 'Unknown'),
            'predicted_phase':    round(float(best_res.x[1]), 2),
            'chi2':               round(float(best_chi2), 4),
        }
    except Exception:
        return None

def superfit_classify(spec, model_list, z):
    results = [optimize_fit(spec, n, z) for n in model_list]
    valid   = [r for r in results if r is not None]
    if not valid:
        return 'FAILED', 'FAILED', np.nan, np.nan
    best = sorted(valid, key=lambda x: x['chi2'])[0]
    return best['predicted_type'], best['predicted_template'], best['predicted_phase'], best['chi2']

# ============================================================
# REBINNING + SIMULATION
# ============================================================

def rebin_spectrum(wavelength, flux, bin_width_aa=10.0):
    wavelength = np.array(wavelength, dtype=float)
    flux       = np.array(flux, dtype=float)
    edges      = np.arange(wavelength[0], wavelength[-1] + bin_width_aa, bin_width_aa)
    ww, ff = [], []
    for i in range(len(edges) - 1):
        m = (wavelength >= edges[i]) & (wavelength < edges[i+1])
        if m.sum() > 0:
            ww.append(0.5*(edges[i]+edges[i+1]))
            ff.append(np.mean(flux[m]))
    return np.array(ww), np.array(ff)

def simulate_one(model_name, z, phase, snr, wave_range, n_wave, bin_width_aa):
    wave = np.linspace(wave_range[0], wave_range[1], n_wave)
    try:
        model  = sncosmo.Model(source=model_name)
        pnames = list(model.param_names)
        model.set(z=z)
        ap    = _amp_param(model_name, model)
        extra = _extra_params(model_name, pnames)
        extra.pop(ap, None)
        try:
            t_min = float(model.mintime())
            t_max = float(model.maxtime())
            phase = 0.0 if (t_max - t_min) < 1.0 else \
                    float(np.clip(phase, t_min + 0.01, t_max - 0.01))
        except Exception:
            pass
        model.set(**extra)
        model.set(**{ap: 1.0, 't0': phase})
        flux_clean = model.flux(0, wave)
    except Exception:
        return None, None
    if not np.any(np.isfinite(flux_clean)) or np.sum(flux_clean > 0) < 10:
        return None, None
    flux_clean = np.maximum(flux_clean, 0)
    med = np.median(flux_clean[flux_clean > 0])
    if med <= 0:
        return None, None
    flux_clean /= med
    flux_noisy = flux_clean + np.random.normal(0, 1.0/snr, n_wave)
    return rebin_spectrum(wave, flux_noisy, bin_width_aa)

# ============================================================
# NGSF RUNNER
# Confirmed working call sequence from live environment.
# ============================================================

def run_ngsf(wave, flux, z, params_path, trial_idx):
    ngsf_dir  = os.path.dirname(os.path.abspath(params_path))
    spec_name = f"trial_{trial_idx:04d}.txt"
    spec_file = os.path.join(ngsf_dir, spec_name)
    np.savetxt(spec_file, np.column_stack((wave, flux)))

    with open(params_path, 'r') as f:
        params = json.load(f)
    params['object_to_fit']     = spec_name
    params['use_exact_z']       = 1
    params['z_exact']           = float(z)
    params['show_plot']         = 0
    params['how_many_plots']    = 0
    params['mask_galaxy_lines'] = 0
    params['mask_telluric']     = 0
    params['lower_lam']         = 0
    params['upper_lam']         = 0
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    original_dir = os.getcwd()
    os.chdir(ngsf_dir)
    try:
        sys.argv[1] = 'parameters.json'
        for mod in list(sys.modules.keys()):
            if mod.startswith('NGSF'):
                del sys.modules[mod]
        from NGSF.sf_class import Superfit
        this_supernova = Superfit()
        this_supernova.sg_error()
        plt.close('all')          # suppress any NGSF diagnostic plots
        this_supernova.superfit()
        plt.close('all')          # suppress NGSF result plots
        results = this_supernova.results
    except Exception as e:
        import traceback
        print(f"\n  NGSF error trial {trial_idx}: {e}")
        traceback.print_exc()
        return 'FAILED', 'FAILED', np.nan, np.nan
    finally:
        os.chdir(original_dir)

    if results is None or len(results) == 0:
        return 'FAILED', 'FAILED', np.nan, np.nan

    top           = results.iloc[0]
    ngsf_type     = parse_ngsf_sn_label(str(top['SN']))
    # Extract template name: "II/2012aw/BC-Ekar phase-band : 5.18B" -> "II/2012aw"
    sn_parts      = str(top['SN']).strip().split('/')
    ngsf_template = '/'.join(sn_parts[:2]) if len(sn_parts) >= 2 else str(top['SN'])
    ngsf_phase    = round(float(top['Phase']), 2)    if pd.notna(top['Phase'])     else np.nan
    ngsf_chi2     = round(float(top['CHI2/dof']), 4) if pd.notna(top['CHI2/dof']) else np.nan
    return ngsf_type, ngsf_template, ngsf_phase, ngsf_chi2

# ============================================================
# MAIN RUN
# ============================================================

def run_comparison():
    model_list_all = list(model_dict.keys())
    model_list_sf  = [m for m in model_list_all if m not in CLASSIFY_EXCLUDE]

    progress_file = os.path.join(RESULTS_DIR, 'comparison_progress.csv')
    np.random.seed(SEED)

    records, start_trial = [], 0
    if os.path.exists(progress_file):
        existing    = pd.read_csv(progress_file)
        records     = existing.to_dict('records')
        start_trial = len(records)
        if start_trial >= N_TRIALS:
            print("Already complete — plotting existing results.")
            plot_comparison(existing)
            return existing
        print(f"Resuming from trial {start_trial}/{N_TRIALS}")
        # Advance RNG to correct state
        for _ in range(start_trial):
            np.random.choice(model_list_all)
            np.random.uniform(*PHASE_RANGE)
            np.random.uniform(*SNR_RANGE)
    else:
        print(f"Starting fresh — {N_TRIALS} trials")

    print(f"Superfit pool : {len(model_list_sf)} templates")
    print(f"Results dir   : {RESULTS_DIR}\n")

    t_start = time.time()

    for trial in tqdm(range(start_trial, N_TRIALS), desc="Trials"):
        model_name = np.random.choice(model_list_all)
        true_phase = round(float(np.random.uniform(*PHASE_RANGE)), 2)
        snr        = round(float(np.random.uniform(*SNR_RANGE)), 1)
        true_type  = model_dict.get(model_name, 'Unknown')

        wave_r, flux_r = simulate_one(
            model_name, Z, true_phase, snr, WAVE_RANGE, N_WAVE, BIN_WIDTH_AA)

        if wave_r is None:
            records.append(dict(
                trial=trial,
                true_template=model_name, true_type=true_type,
                true_phase=true_phase,    snr=snr,
                sf_predicted_template='FAILED',   sf_predicted_type='FAILED',
                sf_predicted_phase=np.nan,        sf_chi2=np.nan,
                ngsf_predicted_template='FAILED', ngsf_predicted_type='FAILED',
                ngsf_predicted_phase=np.nan,      ngsf_chi2=np.nan,
            ))
            pd.DataFrame(records).to_csv(progress_file, index=False)
            continue

        # Superfit
        spec = Spectrum(wave_r, flux_r, redshift=Z)
        sf_type, sf_template, sf_phase, sf_chi2 = superfit_classify(spec, model_list_sf, Z)

        # NGSF
        ngsf_type, ngsf_template, ngsf_phase, ngsf_chi2 = run_ngsf(
            wave_r, flux_r, Z, PARAMS_PATH, trial)

        records.append(dict(
            trial=trial,
            true_template=model_name, true_type=true_type,
            true_phase=true_phase,    snr=snr,
            sf_predicted_template=sf_template,
            sf_predicted_type=sf_type,
            sf_predicted_phase=sf_phase,
            sf_chi2=sf_chi2,
            ngsf_predicted_template=ngsf_template,
            ngsf_predicted_type=ngsf_type,
            ngsf_predicted_phase=ngsf_phase,
            ngsf_chi2=ngsf_chi2,
        ))
        pd.DataFrame(records).to_csv(progress_file, index=False)

        elapsed   = time.time() - t_start
        per_trial = elapsed / (trial - start_trial + 1)
        remaining = per_trial * (N_TRIALS - trial - 1)
        sf_ok   = '✓' if is_correct(true_type, sf_type)   else '✗'
        ngsf_ok = '✓' if is_correct(true_type, ngsf_type) else '✗'
        tqdm.write(
            f"  {trial+1:3d} | true={true_type:8s} ({model_name}) ph={true_phase:+6.1f}d | "
            f"SF {sf_ok} {sf_type:8s} ({sf_template}) ph={str(sf_phase):>6s}d | "
            f"NGSF {ngsf_ok} {ngsf_type:8s} ({ngsf_template}) ph={str(ngsf_phase):>6s}d | "
            f"ETA {remaining/60:.1f}min"
        )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, 'comparison_final.csv'), index=False)
    print(f"\nSaved → {RESULTS_DIR}/comparison_final.csv")
    plot_comparison(df)
    return df

# ============================================================
# PLOT
# ============================================================

TYPE_ORDER = ['Ia','Ia-91bg','Ia-91t','Ia-SC',
              'II','IIn','IIb','II-pec',
              'Ib','Ib/Ic','Ic','Ic-BL','PopIII-II','FAILED']

def build_matrix(df, pred_col):
    present = set(df['true_type'].tolist() + df[pred_col].tolist())
    types   = [t for t in TYPE_ORDER if t in present] + \
              [t for t in present if t not in TYPE_ORDER]
    mat     = pd.DataFrame(0, index=types, columns=types)
    for _, row in df.iterrows():
        t, p = row['true_type'], row[pred_col]
        if t in mat.index and p in mat.columns:
            mat.loc[t, p] += 1
    active = [t for t in types if mat.loc[t].sum() > 0]
    mat    = mat.loc[active, active]
    norm   = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)
    return mat, norm, active

def overall_accuracy(df, pred_col):
    failed  = (df[pred_col] == 'FAILED').sum()
    correct = sum(is_correct(r.true_type, getattr(r, pred_col))
                  for r in df.itertuples())
    return 100 * correct / max(len(df) - failed, 1)

def draw_matrix(ax, mat, norm, active, title, TC, GC, cmap):
    ax.set_facecolor('#0d0d1a')
    for s in ax.spines.values(): s.set_edgecolor(GC)
    n = len(active)
    ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1,
              aspect='auto', interpolation='nearest')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(active, rotation=45, ha='right', fontsize=8, color=TC)
    ax.set_yticklabels(active, fontsize=8, color=TC)
    ax.set_xlabel("Predicted Type", fontsize=10, color=TC, labelpad=6)
    ax.set_ylabel("True Type",      fontsize=10, color=TC, labelpad=6)
    ax.tick_params(colors=TC)
    ax.set_title(title, color='#ffd700', fontsize=12, fontweight='bold', pad=8)
    for i in range(n):
        for j in range(n):
            if is_correct(active[i], active[j]):
                ax.add_patch(plt.Rectangle(
                    (j-.5, i-.5), 1, 1, fill=False,
                    edgecolor='#ffd700',
                    lw=1.5, linestyle='-' if i==j else '--', zorder=3))
    for r in range(n):
        for c in range(n):
            v, ct = norm.iloc[r,c], int(mat.iloc[r,c])
            if ct > 0:
                ax.text(c, r, f"{v:.2f}\n({ct})",
                        ha='center', va='center', fontsize=6.5, zorder=4,
                        color='white' if v>0.45 else '#bbbbbb',
                        fontweight='bold' if is_correct(active[r],active[c]) else 'normal')

def plot_comparison(df):
    sf_mat,   sf_norm,   sf_active   = build_matrix(df, 'sf_predicted_type')
    ngsf_mat, ngsf_norm, ngsf_active = build_matrix(df, 'ngsf_predicted_type')
    sf_acc   = overall_accuracy(df, 'sf_predicted_type')
    ngsf_acc = overall_accuracy(df, 'ngsf_predicted_type')

    cmap = LinearSegmentedColormap.from_list(
        'sn', ['#f7fbff','#4292c6','#08306b'], N=256)
    TC, GC, AC = '#e8e8f0', '#2a2a4a', '#7eb8f7'
    SF_COL   = '#7eb8f7'   # blue  for Superfit
    NGSF_COL = '#ffd700'   # gold  for NGSF

    # ── Figure 1: side-by-side confusion matrices ────────────────────────
    fig1 = plt.figure(figsize=(22, 10), facecolor='#1a1a2e')
    gs1  = gridspec.GridSpec(1, 2, wspace=0.28,
                             left=0.06, right=0.95, top=0.88, bottom=0.12)
    ax_sf   = fig1.add_subplot(gs1[0])
    ax_ngsf = fig1.add_subplot(gs1[1])

    draw_matrix(ax_sf,   sf_mat,   sf_norm,   sf_active,
                f"Superfit  —  {sf_acc:.1f}% overall",  TC, GC, cmap)
    draw_matrix(ax_ngsf, ngsf_mat, ngsf_norm, ngsf_active,
                f"NGSF  —  {ngsf_acc:.1f}% overall",    TC, GC, cmap)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm.set_array([])
    cb = fig1.colorbar(sm, ax=[ax_sf, ax_ngsf], fraction=0.015, pad=0.02)
    cb.set_label('Fraction of true type', color=TC, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TC)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC, fontsize=8)
    fig1.text(0.5, 0.94,
              "Superfit vs NGSF  —  Direct Comparison on Identical Simulated Spectra",
              ha='center', fontsize=14, color=TC, fontweight='bold')
    fig1.text(0.5, 0.915,
              f"{len(df)} trials  |  z={Z}  |  "
              f"phase~U({PHASE_RANGE[0]},{PHASE_RANGE[1]})d  |  "
              f"SNR~U{SNR_RANGE}  |  Ib/Ic soft-match  |  IIP+IIL→II",
              ha='center', fontsize=9, color='#888899')
    out1 = os.path.join(RESULTS_DIR, 'comparison_matrix.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.show()

    # ── Figure 2: phase recovery + SNR curve + per-type bars ─────────────
    fig2 = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    gs2  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                             left=0.08, right=0.96, top=0.91, bottom=0.08)
    ax_phase_sf   = fig2.add_subplot(gs2[0, 0])
    ax_phase_ngsf = fig2.add_subplot(gs2[0, 1])
    ax_snr        = fig2.add_subplot(gs2[1, 0])
    ax_bars       = fig2.add_subplot(gs2[1, 1])

    for ax in [ax_phase_sf, ax_phase_ngsf, ax_snr, ax_bars]:
        ax.set_facecolor('#0d0d1a')
        for s in ax.spines.values(): s.set_edgecolor(GC)
        ax.tick_params(colors=TC)

    # helper: phase limit
    ph_lim = (PHASE_RANGE[0] - 3, PHASE_RANGE[1] + 3)

    # ── Phase recovery: Superfit ─────────────────────────────────────────
    phase_df = df.dropna(subset=['true_phase', 'sf_predicted_phase'])
    tp  = phase_df['true_phase'].values
    sfp = phase_df['sf_predicted_phase'].values
    correct_sf = [is_correct(r.true_type, r.sf_predicted_type)
                  for r in phase_df.itertuples()]
    colors_sf = [SF_COL if c else '#e05555' for c in correct_sf]
    ax_phase_sf.scatter(tp, sfp, c=colors_sf, s=40, alpha=0.8, zorder=3)
    ax_phase_sf.plot(ph_lim, ph_lim, color='#ffd700', lw=1, ls='--', alpha=0.6)
    if len(tp) > 1:
        resid_sf = np.std(sfp - tp)
        ax_phase_sf.text(0.05, 0.93, f"σ = {resid_sf:.1f}d",
                         transform=ax_phase_sf.transAxes,
                         fontsize=9, color=TC)
    ax_phase_sf.set_xlim(*ph_lim); ax_phase_sf.set_ylim(*ph_lim)
    ax_phase_sf.set_xlabel("True Phase (d)",      fontsize=9, color=TC)
    ax_phase_sf.set_ylabel("Predicted Phase (d)", fontsize=9, color=TC)
    ax_phase_sf.set_title("Superfit — Phase Recovery",
                          fontsize=10, color=SF_COL, fontweight='bold', pad=5)
    ax_phase_sf.grid(color=GC, lw=0.5, alpha=0.5)

    # ── Phase recovery: NGSF ────────────────────────────────────────────
    phase_df2 = df.dropna(subset=['true_phase', 'ngsf_predicted_phase'])
    tp2   = phase_df2['true_phase'].values
    ngsfp = phase_df2['ngsf_predicted_phase'].values
    correct_ngsf = [is_correct(r.true_type, r.ngsf_predicted_type)
                    for r in phase_df2.itertuples()]
    colors_ngsf = [NGSF_COL if c else '#e05555' for c in correct_ngsf]
    ax_phase_ngsf.scatter(tp2, ngsfp, c=colors_ngsf, s=40, alpha=0.8, zorder=3)
    ax_phase_ngsf.plot(ph_lim, ph_lim, color='#ffd700', lw=1, ls='--', alpha=0.6)
    if len(tp2) > 1:
        resid_ngsf = np.std(ngsfp - tp2)
        ax_phase_ngsf.text(0.05, 0.93, f"σ = {resid_ngsf:.1f}d",
                           transform=ax_phase_ngsf.transAxes,
                           fontsize=9, color=TC)
    ax_phase_ngsf.set_xlim(*ph_lim); ax_phase_ngsf.set_ylim(*ph_lim)
    ax_phase_ngsf.set_xlabel("True Phase (d)",      fontsize=9, color=TC)
    ax_phase_ngsf.set_ylabel("Predicted Phase (d)", fontsize=9, color=TC)
    ax_phase_ngsf.set_title("NGSF — Phase Recovery",
                            fontsize=10, color=NGSF_COL, fontweight='bold', pad=5)
    ax_phase_ngsf.grid(color=GC, lw=0.5, alpha=0.5)

    # ── SNR vs accuracy ──────────────────────────────────────────────────
    snr_edges = np.linspace(SNR_RANGE[0], SNR_RANGE[1], 6)
    for pred_col, label, col in [
        ('sf_predicted_type',   'Superfit', SF_COL),
        ('ngsf_predicted_type', 'NGSF',     NGSF_COL),
    ]:
        bl, ba, be = [], [], []
        for i in range(len(snr_edges) - 1):
            sub = df[(df['snr'] >= snr_edges[i]) & (df['snr'] < snr_edges[i+1])]
            if len(sub) == 0: continue
            hits = sum(is_correct(r.true_type, getattr(r, pred_col))
                       for r in sub.itertuples())
            acc  = 100 * hits / len(sub)
            bl.append(f"{snr_edges[i]:.0f}–{snr_edges[i+1]:.0f}")
            ba.append(acc)
            be.append(np.sqrt(acc * (100 - acc) / max(len(sub), 1)))
        if bl:
            xs = range(len(bl))
            ax_snr.plot(xs, ba, color=col, lw=2, marker='o', markersize=6,
                        markerfacecolor=col, label=label, zorder=3)
            ax_snr.fill_between(xs, [a-e for a,e in zip(ba,be)],
                                    [a+e for a,e in zip(ba,be)],
                                color=col, alpha=0.12)
            ax_snr.set_xticks(xs)
            ax_snr.set_xticklabels(bl, fontsize=8, color=TC)

    ax_snr.axhline(80, color='#ffd700', lw=0.8, ls='--', alpha=0.6)
    ax_snr.set_ylim(0, 105)
    ax_snr.set_xlabel("SNR bin",      fontsize=9, color=TC)
    ax_snr.set_ylabel("Accuracy (%)", fontsize=9, color=TC)
    ax_snr.set_title("Accuracy vs SNR", fontsize=10, color=AC,
                     fontweight='bold', pad=5)
    ax_snr.legend(fontsize=8, facecolor='#1a1a2e', labelcolor=TC,
                  edgecolor=GC)
    ax_snr.grid(color=GC, lw=0.5, alpha=0.5)

    # ── Per-type accuracy bars ───────────────────────────────────────────
    types_present = [t for t in TYPE_ORDER if t in df['true_type'].values]
    sf_accs, ngsf_accs, ns = [], [], []
    for t in types_present:
        sub = df[df['true_type'] == t]
        ns.append(len(sub))
        sf_accs.append(100 * sum(is_correct(r.true_type, r.sf_predicted_type)
                                 for r in sub.itertuples()) / max(len(sub), 1))
        ngsf_accs.append(100 * sum(is_correct(r.true_type, r.ngsf_predicted_type)
                                   for r in sub.itertuples()) / max(len(sub), 1))

    y   = np.arange(len(types_present))
    h   = 0.35
    ax_bars.barh(y + h/2, sf_accs,   height=h, color=SF_COL,   label='Superfit', alpha=0.85)
    ax_bars.barh(y - h/2, ngsf_accs, height=h, color=NGSF_COL, label='NGSF',     alpha=0.85)
    for i, (sa, na, n) in enumerate(zip(sf_accs, ngsf_accs, ns)):
        ax_bars.text(min(sa+1,99), y[i]+h/2, f"{sa:.0f}%", va='center',
                     fontsize=6.5, color=TC)
        ax_bars.text(min(na+1,99), y[i]-h/2, f"{na:.0f}%", va='center',
                     fontsize=6.5, color=TC)
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels([f"{t}  (n={n})" for t,n in
                             zip(types_present, ns)], fontsize=8, color=TC)
    ax_bars.set_xlim(0, 115)
    ax_bars.axvline(80, color='#ffd700', lw=0.8, ls='--', alpha=0.6)
    ax_bars.set_xlabel("Accuracy (%)", fontsize=9, color=TC)
    ax_bars.set_title("Per-type Accuracy", fontsize=10, color=AC,
                      fontweight='bold', pad=5)
    ax_bars.legend(fontsize=8, facecolor='#1a1a2e', labelcolor=TC,
                   edgecolor=GC)
    ax_bars.grid(axis='x', color=GC, lw=0.5, alpha=0.5)

    fig2.text(0.5, 0.95,
              "Superfit vs NGSF  —  Phase Recovery & Performance Breakdown",
              ha='center', fontsize=13, color=TC, fontweight='bold')
    fig2.text(0.5, 0.928,
              f"{len(df)} trials  |  z={Z}  |  "
              f"phase~U({PHASE_RANGE[0]},{PHASE_RANGE[1]})d  |  SNR~U{SNR_RANGE}",
              ha='center', fontsize=9, color='#888899')

    out2 = os.path.join(RESULTS_DIR, 'comparison_analysis.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.show()

    # Console summary
    print("\n── Per-type accuracy ─────────────────────────────────")
    print(f"  {'Type':<12}  {'Superfit':>10}  {'NGSF':>10}  {'n':>5}")
    print("  " + "─"*42)
    for t, sa, na, n in zip(types_present, sf_accs, ngsf_accs, ns):
        print(f"  {t:<12}  {sa:>8.1f}%  {na:>8.1f}%  {n:>5}")
    print("  " + "─"*42)
    print(f"  {'OVERALL':<12}  {sf_acc:>8.1f}%  {ngsf_acc:>8.1f}%")
    print(f"\nPlots saved:\n  {out1}\n  {out2}")

# ============================================================
# EXECUTE
# ============================================================

if __name__ == '__main__':
    df = run_comparison()
