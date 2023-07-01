"""
pregenerate_dataset_variants.py: Generates all possible dataset variants for a
    given dataset. As generating the dataset from scratch can be very time-consuming,
    running this script before starting to train models can save a considerable
    amount of time.
"""
import pandas as pd

from utils import *
from config import *


def generate_datasets():
    """
    generate_datasets: Generates all possible dataset variants for a given dataset.

    :return: None.
    """
    print(f"Generating datasets for {sampling_window_hours}-hour window.")

    # Parameters:
    use_clusters_options = [True, False]
    discrete_action_options = [True, False]
    apache2_rewards_options = [True, False]
    terminal_flags_options = [True, False]

    # Load CSVs:
    print("Reading dataset CSVs...")
    train_set = pd.read_csv(f"../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}train_data_trajectory.csv")
    val_set = pd.read_csv(f"../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}val_data_trajectory.csv")
    test_set = pd.read_csv(f"../data/{dataset_prefix}{sampling_window_hours}hour_window/{dataset_prefix}test_data_trajectory.csv")
    print("Datasets loaded")

    options_list = [use_clusters_options, discrete_action_options, apache2_rewards_options, terminal_flags_options]
    for use_clusters, discrete_actions, apache2_rewards, terminal_flags in itertools.product(*options_list):
        # The dataframe_to_dataset function automatically saves the dataset to
        # the specified path, or loads the dataset from the specified path if
        # it already exists. Therefore, this script can be run multiple times
        # or in batches with minimal unnecessary re-computation.
        train_dataset = dataframe_to_dataset(
            train_set,
            use_clusters=use_clusters,
            discrete_action=discrete_actions,
            apache2_rewards=apache2_rewards,
            terminal_flags=terminal_flags,
            save_path_base=f"../data/{dataset_prefix}{sampling_window_hours}hour_window/generated_datasets/Train"
        )
        f"../data/{sampling_window_hours}hour_window/train_data_trajectory.csv"
        val_dataset = dataframe_to_dataset(
            val_set,
            use_clusters=use_clusters,
            discrete_action=discrete_actions,
            apache2_rewards=apache2_rewards,
            terminal_flags=terminal_flags,
            save_path_base=f"../data/{dataset_prefix}{sampling_window_hours}hour_window/generated_datasets/Val"
        )
        test_dataset = dataframe_to_dataset(
            test_set,
            use_clusters=use_clusters,
            discrete_action=discrete_actions,
            apache2_rewards=apache2_rewards,
            terminal_flags=terminal_flags,
            save_path_base=f"../data/{dataset_prefix}{sampling_window_hours}hour_window/generated_datasets/Test"
        )

    return


if __name__ == '__main__':
    set_seed(42)
    generate_datasets()