"""
config.py: Contains the configuration for the data processing scripts. Among others,
    the most impoorant variables are:
        - sampling_window_hours: The sampling window in hours.
        - dataset_prefix: The prefix of the dataset. "" for MIMIC-IV, "aumcdb_"
            for AmsterdamUMCdb.
        - demographical_vars & observational_vars: the features to be included
            in the dataset.
        - Vt, PEEP, FiO2 bins: The bins for the discretization of the Vt, PEEP,
            and FiO2 settings in the action space.
        - APACHE-II bins: The bins for the discretization of the APACHE-II
            intermediate rewards inspired by Kondrup et al. (2022).
        - GAMMA: Discount factor.
        - REWARD: The reward to be given at the end of the episode if the
            patient is still alive.
"""
# source: https://github.com/florisdenhengst/ventai/blob/main/notebooks/config.py
import numpy as np
sampling_window_hours = 1
dataset_prefix = "" # Default of "" uses MIMIC-IV. "aumcdb_" uses AmsterdamUMCdb.

demographical_vars = [
    'admission_age',
    'adult_ibw',
    'weight',
    'icu_readmission',
    'hospmort',
    'mort90day',
]

observational_vars = [
    'sofa',
    'sirs',
    'gcs',
    'heartrate',
    'sysbp',
    'meanbp',
    'diasbp',
    'shockindex',
    'resprate',
    'resprate_spont',
    'spo2',
    'tempc',
    'potassium',
    'sodium',
    'chloride',
    'glucose',
    'bun',
    'creatinine',
    'magnesium',
    'calcium',
    'ionizedcalcium',
    'bilirubin',
    'albumin',
    'hemoglobin',
    'wbc',
    'platelet',
    'ptt',
    'inr',
    'ph',
    'pao2',
    'paco2',
    'base_excess',
    'bicarbonate',
    'lactate',
    'pao2fio2ratio',
    'iv_total',
    'vaso_total',
    'urineoutput',
    'cum_fluid_balance',
    'peep',
    'fio2',
    'tidal_volume',
    'tidal_volume_spont',
    'mechvent',
    'plateau_pressure',
    'pip',
    'map',
    'crp',
    'etco2',
]

state_vars = demographical_vars + observational_vars

Vt_bins = [0, 2.5, 5, 7.5, 10, 12.5, 15, np.inf]
PEEP_bins = [0, 5, 7, 9, 11, 13, 15, np.inf]
FiO2_bins = [0, 30, 35, 40, 45, 50, 55, np.inf]

# APACHE-II based intermediate rewards.
# Inspired by: https://github.com/FlemmingKondrup/DeepVent
APACHE2_bins = {
    "tempc": [(41, 4), (39, 3), (38.5, 1), (36, 0), (34, 1), (32, 2), (30, 3), (0, 4)],
    "meanbp": [(160, 4), (130, 3), (110, 2), (70, 0), (50, 2), (0, 4)],
    "heartrate": [(180, 4), (140, 3), (110, 2), (70, 0), (55, 2), (40, 3), (0, 4)],
    "ph": [(7.7, 4), (7.6, 3), (7.5, 1), (7.33, 0), (7.25, 2), (7.15, 3), (0, 4)],
    "sodium": [(180, 4), (160, 3), (155, 2), (150, 1), (130, 0), (120, 2), (111, 3), (0, 4)],
    "potassium": [(7, 4), (6, 3), (5.5, 1), (3.5, 0), (3, 1), (2.5, 2), (0, 4)],
    "creatinine": [(305, 4), (170, 3), (130, 2), (53, 0), (0, 2)],
    "wbc": [(40, 4), (20, 2), (15, 1), (3, 0), (1, 2), (0, 4)],
}
APACHE2_bins_labels = {
    "tempc": ([0, 29.9, 31.9, 33.9, 35.9, 38.4, 38.9, 40.9, np.inf], [4, 3, 2, 1, 0, 1, 3, 4]),
    "meanbp": ([0, 49, 69, 109, 129, 159, np.inf], [4, 2, 0, 2, 3, 4]),
    "heartrate": ([0, 39, 54, 69, 109, 139, 179, np.inf], [4, 3, 2, 0, 2, 3, 4]),
    "ph": ([0, 7.15, 7.24, 7.32, 7.49, 7.59, 7.69, np.inf], [4, 3, 2, 0, 1, 3, 4]),
    "sodium": ([0, 110, 119, 129, 149, 154, 159, 179, np.inf], [4, 3, 2, 0, 1, 2, 3, 4]),
    "potassium": ([0, 2.5, 2.9, 3.4, 5.4, 5.9, 6.9, np.inf], [4, 2, 1, 0, 1, 3, 4]),
    "creatinine": ([0, 0.6, 1.4, 1.9, 3.4, np.inf], [2, 0, 2, 3, 4]),
    "wbc": ([0, 1, 2.9, 14.9, 19.9, 39.9, np.inf], [4, 2, 0, 1, 2, 4]),
}

GAMMA = 0.99
REWARD = 100
MAX_GCS_CONTRIBUTION = 12 # Max GCS - Min GCS = 15 - 3 = 12
MAX_APACHE2_SCORE = sum([l[0][1] for l in APACHE2_bins.values()]) + MAX_GCS_CONTRIBUTION
MIN_APACHE2_SCORE = 0
APACHE2_REWARD_SCALING_FACTOR = 100
