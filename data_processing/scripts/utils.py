"""
utils.py: Provides utility functions for data processing.
"""
import os
import io
import sys
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import torch
import itertools
import d3rlpy

import joblib

from config import *

def set_seed(seed):
    """
    set_seed: Sets the seed for numpy and random.

    :param seed: The seed to be set.
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)


def get_state_varnames(to_exclude=['hospmort', 'mort90day', 'icu_readmission']):
    """
    get_state_varnames: Returns the state variables and their names in the table.

    :param to_exclude: The variables to be excluded from the state.
    :return: Tuple of state variables and the corresponding names in the table.
    """
    state_variables = [var for var in state_vars if var not in to_exclude]
    state_varnames = []
    for var in state_variables:
        state_varnames.append(var + '_final')

    return state_variables, state_varnames


def generate_save_name(
        use_clusters,
        discrete_action,
        apache2_rewards,
        terminal_flags,
        save_path_base
    ):
    """
    generate_save_name: Generates a save name for the dataset.

    :param use_clusters: Whether to use state clusters or not.
    :param discrete_action: Whether to use discrete actions or not.
    :param apache2_rewards: Whether to use the Apache-II intermediate rewards or not.
    :param terminal_flags: Whether to use Terminal Flags or not.
    :param save_path_base: The base path to save the dataset to.
    :return: The save name for the dataset.
    """
    cluster_label = "_Clusters" if use_clusters else ""
    action_label = "_DiscA" if discrete_action else "_ContA"
    apache2_label = "_Apache2" if apache2_rewards else ""
    terminal_label = "_TermFlags" if terminal_flags else ""
    return f"{save_path_base}{cluster_label}{action_label}{apache2_label}{terminal_label}.bin"


def create_dataset(states, actions, rewards, terminals, discrete_action, save_path):
    """
    create_dataset: Creates a d3rlpy dataset from the states, actions, rewards, and terminals.

    :param states: List of states.
    :param actions: List of actions.
    :param rewards: List of rewards.
    :param terminals: List of the binary episode terminal labels.
    :param discrete_action: Whether to use discrete actions or not.
    :param save_path: The path to save the dataset to.
    :return: The d3rlpy dataset.
    """
    states, actions, rewards, terminals = np.array(states), np.array(actions), np.array(rewards), np.array(terminals)
    rewards = np.nan_to_num(rewards)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=states,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        discrete_action=discrete_action
    )

    if save_path != None:
        print(f"Saving dataset to {save_path}...")
        joblib.dump(dataset, f"{save_path}", compress=True)
        print(f"Dataset saved successfully!")

    return dataset


def iterative_conversion(df, use_clusters=True, discrete_action=True, apache2_rewards=True, terminal_flags=True, save_path=None):
    """
    iterative_conversion: Converts the dataframe to a d3rlpy dataset iteratively.
        The iterative conversion is used when clusters or terminal flags are applied.
        In both cases, we have to manually insert a custom terminal transition
        at the end of each episode in the dataset.

    :param df: The dataframe to be converted.
    :param use_clusters: Whether to use state clusters or not.
    :param discrete_action: Whether to use discrete actions or not.
    :param apache2_rewards: Whether to use the Apache-II intermediate rewards or not.
    :param terminal_flags: Whether to use Terminal Flags or not.
    :param save_path: The path to save the dataset to.
    :return: The d3rlpy dataset.
    """
    state_variables, state_varnames = get_state_varnames()
    states, actions, rewards, terminals = [], [], [], []
    if not discrete_action:
        df['vt_derived'] = df['vt_derived'].fillna(0.0)
        df['peep_unscaled'] = df['peep_unscaled'].fillna(0.0)
        df['fio2_unscaled'] = df['fio2_unscaled'].fillna(20.0)

    # When using clusters, we do not use terminal flags.
    if use_clusters:
        terminal_flags = False

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if use_clusters:
            state = row.state
        else:
            state = row[state_varnames].copy()

        if terminal_flags:
            state['terminal_int'] = 0
            state['expired_int'] = 0

        states.append(state)

        if discrete_action:
            action = row.action
        else:
            action = [row['vt_derived'], row['peep_unscaled'], row['fio2_unscaled']]

        actions.append(action)

        reward = row.reward
        if apache2_rewards and not row.terminal:
            # The APACHE-II reward is only applied to the intermediate transitions.
            reward += row.apache2_reward
        rewards.append(reward)
        terminals.append(0)

        if row.terminal:
            if use_clusters:
                next_state = row.next_state
            else:
                next_state = row[state_varnames].copy()
                if terminal_flags:
                    next_state['terminal_int'] = 1
                    next_state['expired_int'] = 1 if row.reward < 0 else 0
            states.append(next_state if use_clusters else next_state.values)
            actions.append(action) # In order: Vt, PEEP, FiO2
            rewards.append(0)
            terminals.append(1)

    final_states = np.array(states) if not use_clusters else np.eye(652)[np.array(states)]
    return create_dataset(final_states, actions, rewards, terminals, discrete_action, save_path)


def dataframe_to_dataset(df, use_clusters=True, discrete_action=True, apache2_rewards=True, terminal_flags=True, save_path_base=None, print_vars=True):
    """
    dataframe_to_dataset: Converts the dataframe to a d3rlpy dataset.

    :param df: The dataframe to be converted.
    :param use_clusters: Whether to use state clusters or not.
    :param discrete_action: Whether to use discrete actions or not.
    :param apache2_rewards: Whether to use the Apache-II intermediate rewards or not.
    :param terminal_flags: Whether to use Terminal Flags or not.
    :param save_path_base: The base path to save the dataset to.
    :param print_vars: Whether to print the state variables or not.
    :return: The d3rlpy dataset.
    """
    save_path = generate_save_name(use_clusters, discrete_action, apache2_rewards, terminal_flags, save_path_base)
    state_variables, state_varnames = get_state_varnames()

    if print_vars:
        print("Dataset State variables: ", state_variables)

    # The generated datasets are saved to disk to avoid having to re-generate
    # them every time we want to use them.
    if os.path.isfile(save_path):
        print(f"Previously generated dataset found at {save_path}! Loading...")
        return joblib.load(save_path)

    df['action'] = df['action'].astype(int)
    df['terminal_int'] = df['terminal'].astype(int)

    # When using state clusters or Terminal Flags, we have to manually insert
    # a custom terminal transition at the end of each episode in the dataset.
    # Therefore, we cannot easily convert the dataframe to a dataset directly.
    # Instead, we have to do it iteratively.
    if use_clusters or terminal_flags:
        return iterative_conversion(df, use_clusters, discrete_action, apache2_rewards, terminal_flags, save_path)

    states = df[state_varnames].values

    if discrete_action:
        actions = df['action'].values
    else:
        actions = df[['vt_derived', 'peep_unscaled', 'fio2_unscaled']].values

    if apache2_rewards:
        rewards = df['reward'].fillna(0).values + df['apache2_reward'].fillna(0).values
    else:
        rewards = df['reward'].fillna(0).values

    terminals = df['terminal_int'].values

    return create_dataset(states, actions, rewards, terminals, discrete_action, save_path)


