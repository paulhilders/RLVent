"""
dataset_extraction.py: Augments processed datasets with RL-specific information.
"""
import os
import io
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from config import *


def load_state_varnames(dataset):
    """
    load_state_varnames: Loads state variable names from dataset.

    :param dataset: Pandas DataFrame containing the dataset.
    :return: List of state variable names and corresponding column labels in
        the dataframe.
    """
    state_vars = []
    state_vars_final = []
    obs_state_vars_final = []
    demo_state_vars_final = []
    for var in dataset.columns:
        if var[-6:] == "_final":
            state_vars_final.append(var)
            state_vars.append(var[:-6])

            if var[:-6] in observational_vars:
                obs_state_vars_final.append(var)
            elif var[:-6] in demographical_vars:
                demo_state_vars_final.append(var)
    print(state_vars)
    print(state_vars_final)
    print(obs_state_vars_final)
    print(demo_state_vars_final)
    print(state_vars_final == (obs_state_vars_final + demo_state_vars_final))
    return state_vars, state_vars_final, obs_state_vars_final, demo_state_vars_final


def add_kmeans_clusters(train_set, val_set, test_set, state_vars_final, n_states=650):
    """
    add_kmeans_clusters: Adds K-Means clusters to the dataset, as described in
        the VentAI paper by Peine et al. (2021).

    :param train_set: Pandas DataFrame containing the training set.
    :param val_set: Pandas DataFrame containing the validation set.
    :param test_set: Pandas DataFrame containing the test set.
    :param state_vars_final: List of state variable names that are included in
        our feature set.
    :param n_states: Number of state clusters. Default: 650.
    :return: The training-, validation-, and test datasets containing the state
        clusters.
    """
    if os.path.exists(f"../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}KMeansclusterer.bin"):
        print("*** Loading previously trained Clusterer")
        clusterer = joblib.load(f"../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}KMeansclusterer.bin")
    else:
        print("*** Fitting K-Means to States")
        clusterer = MiniBatchKMeans(n_clusters=n_states,
                                    batch_size=1024,
                                    n_init=10,
                                    max_no_improvement=10,
                                    verbose=1).fit(train_set[state_vars_final])

    print("*** Using K-Means to map states to clusters ***")
    train_clusters = clusterer.predict(train_set[state_vars_final])
    val_clusters = clusterer.predict(val_set[state_vars_final])
    test_clusters = clusterer.predict(test_set[state_vars_final])

    # Save KMeans model
    joblib.dump(clusterer, f"../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}KMeansclusterer.bin", compress=True)

    # Add states to tables:
    train_clusters_series = pd.Series(train_clusters)
    val_clusters_series = pd.Series(val_clusters)
    test_clusters_series = pd.Series(test_clusters)

    train_set['state'] = train_clusters_series
    val_set['state'] = val_clusters_series
    test_set['state'] = test_clusters_series

    return train_set, val_set, test_set


def discretize_actions(dataset, levels=7):
    """
    discretize_actions: Discretizes the action space of the dataset using the
        value bins defined in the config.py file.

    :param dataset: Pandas DataFrame containing the dataset.
    :param levels: Number of levels to discretize the action space into. Default: 7.
    :return: The dataset with discretized actions.
    """
    # Derive Ideal body weight adjusted tidal volume.
    dataset['vt_derived'] = dataset['tidal_volume_unscaled'] / dataset['adult_ibw_unscaled']

    dataset["vt_disc"] = pd.cut(dataset.vt_derived, Vt_bins, labels=False, retbins=True)[0]
    dataset["vt_disc"] = dataset["vt_disc"].fillna(0).astype(int)

    dataset["peep_disc"] = pd.cut(dataset.peep_unscaled, PEEP_bins, labels=False, retbins=True)[0]
    dataset["peep_disc"] = dataset["peep_disc"].fillna(0).astype(int)

    dataset["fio2_disc"] = pd.cut(dataset.fio2_unscaled, FiO2_bins, labels=False, retbins=True)[0]
    dataset["fio2_disc"] = dataset["fio2_disc"].fillna(0).astype(int)

    # Discrete action = Vt * levels^2 + PEEP * levels + FiO2
    dataset["action"] = dataset.vt_disc * (levels) ** 2 + dataset.peep_disc * (levels) + dataset.fio2_disc
    return dataset


def add_trajectory_info(dataset, terminal_mort="650", terminal_nonmort="651"):
    """
    add_trajectory_info: Adds terminal labels, state-action labels, and next
        state labels to the dataset. In addition, the terminal states are set.

    :param dataset: Pandas DataFrame containing the dataset.
    :param terminal_mort: Cluster ID used to denote the terminal state where
        the patient dies. Default: 650.
    :param terminal_nonmort: Cluster ID used to denote the terminal state where
        the patient survives. Default: 651.
    :return: The dataset, augmented with trajectory information.
    """
    dataset['terminal'] = False
    dataset.loc[dataset.groupby('unique_id').tail(1).index, 'terminal'] = True

    dataset['state_action'] = dataset.agg('{0[state]}-{0[action]}'.format, axis=1)
    dataset['next_state'] = dataset.state.astype(str).shift(-1)

    dataset.loc[dataset.terminal & ((dataset.hospmort == "t") | (dataset.mort90day == 't')), 'next_state'] = terminal_mort
    dataset.loc[dataset.terminal & ((dataset.mort90day == 'f') & (dataset.hospmort == "f")), 'next_state'] = terminal_nonmort
    return dataset


def add_rewards(dataset, terminal_mort="650", terminal_nonmort="651"):
    """
    add_rewards: Adds rewards to the dataset. The rewards are based on the
        terminal states and the APACHE II Score (if applicable).

    :param dataset: Pandas DataFrame containing the dataset.
    :param terminal_mort: Cluster ID used to denote the terminal state where
        the patient dies. Default: 650.
    :param terminal_nonmort: Cluster ID used to denote the terminal state where
        the patient survives. Default: 651.
    :return: The dataset, augmented with rewards.
    """
    # Add hospital / 90-day mortality rewards
    dataset['reward'] = 0
    dataset.loc[dataset.next_state == terminal_mort, 'reward'] = -REWARD
    dataset.loc[dataset.next_state == terminal_nonmort, 'reward'] = REWARD

    # Add intermediate rewards based on a modified version of the APACHE II Score.
    # Inspired by: https://github.com/FlemmingKondrup/DeepVent/blob/main/utils/compute_trajectories_utils.py
    for var in APACHE2_bins_labels.keys():
        bins = APACHE2_bins_labels[var][0]
        labels = APACHE2_bins_labels[var][1]
        dataset[f"{var}_temp"] = dataset[f"{var}_unscaled"]
        scores = pd.cut(dataset[f"{var}_unscaled"], bins, labels=labels, ordered=False)
        dataset[f"{var}_apache2score"] = scores

    dataset["apache2score"] = sum([dataset[f"{var}_apache2score"].astype('float').astype('Int32') for var in APACHE2_bins_labels.keys()])
    dataset["apache2score"] = dataset["apache2score"] + (15 - dataset["gcs_unscaled"])
    dataset['next_apache2score'] = dataset.apache2score.astype('Int64').shift(-1)

    normalization_factor = MAX_APACHE2_SCORE - MIN_APACHE2_SCORE
    dataset.loc[dataset.terminal == False, 'apache2_reward'] = APACHE2_REWARD_SCALING_FACTOR * (dataset["apache2score"] - dataset['next_apache2score']) / normalization_factor
    return dataset



def extract_rl_datasets():
    """
    extract_rl_datasets: Extracts the reinforcement learning datasets from the
        observational dataset. The datasets are saved to the data folder.

    :return: None.
    """
    print(f"*** Extracting Reinforcement learning datasets with a {sampling_window_hours}-hour window ***")

    # Step 1: Load datasets.
    print("Step 1: Loading datasets...")
    train_set = pd.read_csv(f'../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}train_data.csv')
    val_set = pd.read_csv(f'../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}val_data.csv')
    test_set = pd.read_csv(f'../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}test_data.csv')

    scaled_observational_varnames = joblib.load(f'../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}scaled_observational_varnames.bin')
    observational_vars_scaler = joblib.load(f"../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}observational_vars_scaler.bin")
    scaled_demographical_varnames = joblib.load(f'../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}scaled_demographical_varnames.bin')
    demographical_vars_scaler = joblib.load(f"../data_processors/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}demographical_vars_scaler.bin")

    state_vars, state_vars_final, obs_state_vars_final, demo_state_vars_final = load_state_varnames(train_set)
    print("Step 1 complete: Dataframes loaded successfully!\n")

    # Step 2: VentAI-like state clustering using KMeans
    print("Step 2: Clustering states with KMeans...")
    train_set, val_set, test_set = add_kmeans_clusters(train_set, val_set, test_set, state_vars_final)
    print('Step 2 completed: Added clustered states successfully!\n')

    # Step 3: Unscale data
    print("Step 3: Unscaling data...")
    obs_unscaled_vars = [var + "_unscaled" for var in state_vars if var in observational_vars]
    demo_unscaled_vars = [var + "_unscaled" for var in state_vars if var in demographical_vars]
    unscaled_vars = [var + "_unscaled" for var in state_vars]

    train_set[demo_unscaled_vars] = demographical_vars_scaler.inverse_transform(train_set[demo_state_vars_final])
    train_set[obs_unscaled_vars] = observational_vars_scaler.inverse_transform(train_set[obs_state_vars_final])

    val_set[demo_unscaled_vars] = demographical_vars_scaler.inverse_transform(val_set[demo_state_vars_final])
    val_set[obs_unscaled_vars] = observational_vars_scaler.inverse_transform(val_set[obs_state_vars_final])

    test_set[demo_unscaled_vars] = demographical_vars_scaler.inverse_transform(test_set[demo_state_vars_final])
    test_set[obs_unscaled_vars] = observational_vars_scaler.inverse_transform(test_set[obs_state_vars_final])
    print("Step 3 completed!\n")

    # Step 4: Discretize Actions in VentAI-fashion
    print("Step 4: Discretizing Actions in VentAI-fashion...")
    train_set = discretize_actions(train_set)
    val_set = discretize_actions(val_set)
    test_set = discretize_actions(test_set)
    print("Step 4 completed!\n")

    # Step 5: Add trajectory information
    print("Step 5: Adding trajectory information...")
    train_set = add_trajectory_info(train_set)
    val_set = add_trajectory_info(val_set)
    test_set = add_trajectory_info(test_set)
    print("Step 5 completed!\n")

    # Step 6: Add rewards
    print("Step 6: Adding rewards...")
    train_set = add_rewards(train_set)
    val_set = add_rewards(val_set)
    test_set = add_rewards(test_set)
    print(train_set.columns)
    print(train_set)
    print("Step 6 completed!\n")

    # Step 7: Save final datasets
    print("Step 7: Saving final datasets")
    print("Saving training data...")
    train_set.to_csv(f"../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}train_data_trajectory.csv")
    print("Saving validation data...")
    val_set.to_csv(f"../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}val_data_trajectory.csv")
    print("Saving test data...")
    test_set.to_csv(f"../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}test_data_trajectory.csv")
    print("Step 7 completed!\n")

if __name__ == '__main__':
    set_seed(42)
    extract_rl_datasets()