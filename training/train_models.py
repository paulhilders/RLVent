"""
train_models.py: Provides function for the training of models.
"""
import os
import io
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torch
import itertools

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import d3rlpy

from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.preprocessing import MinMaxRewardScaler

from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.encoders import VectorEncoderFactory

from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discrete_action_match_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer

from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_prediction_variance_scorer

sys.path.insert(0, '../data_processing/scripts/')
from utils import *
from config import *

from training_config import *
from evaluation import *


def print_settings_summary():
    """
    print_settings_summary: Prints a summary of the hyperparameter configurations.

    :return: None
    """
    print("*************************************")
    print("HYPERPARAMETER CONFIGURATIONS:")
    print(f"-> Dataset prefix = {dataset_prefix}")
    print(f"-> N-epochs = {n_epochs}")
    print(f"-> N_steps_per_epoch = {n_steps_per_epoch}")
    print(f"-> N-epochs (evaluation) = {n_epochs_eval}")
    print(f"-> N_steps_per_epoch (evaluation) = {n_steps_per_epoch_eval}")
    print(f"-> algo_label = {algo_label}")
    print(f"-> tensorboard_dir = {tensorboard_dir}")
    print(f"-> use_clusters_options = {use_clusters_options}")
    print(f"-> discrete_action_options = {discrete_action_options}")
    print(f"-> apache2_rewards_options = {apache2_rewards_options}")
    print(f"-> terminal_flags_options = {terminal_flags_options}")
    print(f"-> use_encoder_options = {use_encoder_options}")
    print(f"-> distQ_options = {distQ_options}")
    print(f"-> train_eval_options = {train_eval_options}")
    print(f"-> use_batch_norm_options = {use_batch_norm_options}")
    print(f"-> use_dropout_options = {use_dropout_options}")
    print(f"-> use_dense_options = {use_dense_options}")
    print(f"-> hidden_units_options = {hidden_units_options}")
    print(f"-> n_frames_options = {n_frames_options}")
    print(f"-> n_steps_options = {n_steps_options}")
    print(f"-> n_critics_options = {n_critics_options}")
    print("*************************************")


def load_settings(verbose=True):
    """
    load_settings: Loads the hyperparameter configurations from the training_config.py file.

    :param verbose: Whether to print a summary of the hyperparameter configurations.
    :return: A list of all unique combinations of settings.
    """
    if verbose:
        print_settings_summary()

    # Generate a list of all unique combinations of settings.
    setting_combinations = list(dict.fromkeys(list(itertools.product(
        use_clusters_options,
        discrete_action_options,
        apache2_rewards_options,
        terminal_flags_options,
        use_encoder_options,
        distQ_options,
        train_eval_options,
        use_batch_norm_options,
        use_dropout_options,
        use_dense_options,
        hidden_units_options,
        n_frames_options,
        n_steps_options,
        n_critics_options
    ))))
    return setting_combinations


def read_data_csvs(path=f"../data_processing/data/{dataset_prefix}{sampling_window_hours}hour_window/"):
    """
    read_data_csvs: Reads the dataset CSVs.

    :param path: The path to the dataset CSVs.
    :return: The train, validation and test datasets.
    """
    print("Reading dataset CSVs...")
    train_set = pd.read_csv(f"{path}{dataset_prefix}train_data_trajectory.csv")
    val_set = pd.read_csv(f"{path}{dataset_prefix}val_data_trajectory.csv")
    test_set = pd.read_csv(f"{path}{dataset_prefix}test_data_trajectory.csv")
    print("CSVs loaded successfully!\n")
    return train_set, val_set, test_set


def initialize_algo(algo_label):
    """
    initialize_algo: Initializes the algorithm.

    :param algo_label: The algorithm label. (e.g. "DiscreteCQL")
    :return: The algorithm.
    """
    if algo_label == "DiscreteCQL":
        return d3rlpy.algos.DiscreteCQL
    elif algo_label == "CQL":
        return d3rlpy.algos.CQL
    elif algo_label == "NFQ":
        return d3rlpy.algos.NFQ
    elif algo_label == "DQN":
        return d3rlpy.algos.DQN
    elif algo_label == "DoubleDQN":
        return d3rlpy.algos.DoubleDQN
    elif algo_label == "COMBO":
        return d3rlpy.algos.COMBO
    elif algo_label == "DiscreteCOMBO":
        return d3rlpy.algos.DiscreteCOMBO
    elif algo_label == "DiscreteCOUPLe":
        return d3rlpy.algos.DiscreteCOUPLe
    elif algo_label == "Dynamics":
        return d3rlpy.dynamics.ProbabilisticEnsembleDynamics
    else:
        raise Exception(f"Chosen d3rlpy algorithm \'{algo_label}\' is unkown! Please select a valid algorithm.")


def generate_experiment_name(
        use_clusters,
        discrete_action,
        apache2_rewards,
        terminal_flags,
        use_encoder,
        distQ,
        algo_label,
        n_epochs,
        n_steps,
        n_frames,
        n_critics,
        hidden_units,
        train_eval,
        use_batch_norm,
        use_dropout,
        use_dense,
        experiment_label=""
    ):
    """
    generate_experiment_name: Generates the experiment name.

    :param use_clusters: Whether to use clusters.
    :param discrete_action: Whether to use discrete actions.
    :param apache2_rewards: Whether to use Apache2 rewards.
    :param terminal_flags: Whether to use terminal flags.
    :param use_encoder: Whether to use an encoder.
    :param distQ: Whether to use a distributional Q-function.
    :param algo_label: The algorithm label (e.g. "DiscreteCQL").
    :param n_epochs: The number of epochs.
    :param n_steps: The number of look-ahead steps for N-step temporal
        difference learning.
    :param n_frames: The number of frames (primarily useful for image inputs).
    :param n_critics: The number of critics.
    :param hidden_units: List of integers denoting the sizes of the hidden layers.
    :param train_eval: Whether to use the training dataset instead of the
        validation dataset during evaluation.
    :param use_batch_norm: Whether to use batch normalization.
    :param use_dropout: Whether to use dropout.
    :param use_dense: Whether to use dense layers.
    :param experiment_label: The experiment label (e.g. "Test").
    :return: The experiment name.
    """
    cluster_label = "_Clusters" if use_clusters else ""
    action_label = "_DiscA" if discrete_action else "_ContA"
    apache2_label = "_Apache2" if apache2_rewards else ""
    terminal_label = "_TermFlags" if terminal_flags else ""

    bn_label = "(BN)" if use_batch_norm else ""
    dropout_label = "(Dropout)" if use_dropout else ""
    dense_label = "(Dense)" if use_dense else ""

    encoder_label = f"_Encoded[{','.join(str(layer) for layer in hidden_units)}]{bn_label}{dropout_label}{dense_label}" if use_encoder else ""
    distQ_label = "_distQ" if distQ else ""
    n_steps_label = f"_{n_steps}steps" if n_steps > 1 else ""
    n_frames_label = f"_{n_frames}frames" if n_frames > 1 else ""
    n_critics_label = f"_{n_critics}critics" if n_critics > 1 else ""
    train_eval_label = "_TrainEval" if train_eval else "_ValEval"

    experiment_name = f"{algo_label}_{experiment_label}_{n_epochs}epochs{n_steps_label}{n_frames_label}{n_critics_label}{cluster_label}{action_label}{apache2_label}{terminal_label}{encoder_label}{distQ_label}{train_eval_label}"

    return experiment_name


def load_data(train_set, val_set, test_set, use_clusters, discrete_action, apache2_rewards, terminal_flags, path="./data/generated_datasets/"):
    """
    load_data: Loads the data.

    :param train_set: The training dataset.
    :param val_set: The validation dataset.
    :param test_set: The test dataset.
    :param use_clusters: Whether to use clusters.
    :param discrete_action: Whether to use discrete actions.
    :param apache2_rewards: Whether to use Apache2 rewards.
    :param terminal_flags: Whether to use terminal flags.
    :param path: The path to the data.
    :return: The d3rlpy training, validation and test MDPDatasets.
    """
    # Load data
    train_dataset = dataframe_to_dataset(
        train_set,
        use_clusters=use_clusters,
        discrete_action=discrete_action,
        apache2_rewards=apache2_rewards,
        terminal_flags=terminal_flags,
        save_path_base=f"{path}Train"
    )
    val_dataset = dataframe_to_dataset(
        val_set,
        use_clusters=use_clusters,
        discrete_action=discrete_action,
        apache2_rewards=apache2_rewards,
        terminal_flags=terminal_flags,
        save_path_base=f"{path}Val"
    )
    test_dataset = dataframe_to_dataset(
        test_set,
        use_clusters=use_clusters,
        discrete_action=discrete_action,
        apache2_rewards=apache2_rewards,
        terminal_flags=terminal_flags,
        save_path_base=f"{path}Test"
    )
    return train_dataset, val_dataset, test_dataset


def load_scalers(discrete_action):
    """
    load_scalers: Loads the action and reward scalers.

    :param discrete_action: Whether to use discrete actions.
    :return: The action and reward scalers.
    """
    # Actions only have to be scaled in the continuous control setting.
    if discrete_action:
        action_scaler = None
    else:
        action_scaler = MinMaxActionScaler(minimum=[0.0, 0.0, 20.0], maximum=[28.0, 50.0, 100.0])

    reward_scaler = MinMaxRewardScaler(minimum=-REWARD, maximum=REWARD)
    return action_scaler, reward_scaler


def train_model(
        algo_label="DiscreteCQL",
        use_clusters=False,
        discrete_action=True,
        apache2_rewards=False,
        terminal_flags=False,
        use_encoder=False,
        distQ=False,
        n_frames=1,
        n_steps=1,
        n_critics=1,
        hidden_units = [512, 512, 256, 256],
        train_eval=False,
        use_batch_norm=True,
        use_dropout=True,
        use_dense=True,
        tensorboard_dir="tensorboard_log",
        fit_evaluator=True
    ):
    """
    train_model: Trains a model with the specified training configuration.

    :param algo_label: The algorithm label. Default: "DiscreteCQL".
    :param use_clusters: Whether to use clusters. Default: False.
    :param discrete_action: Whether to use discrete actions. Default: True.
    :param apache2_rewards: Whether to use Apache2 rewards. Default: False.
    :param terminal_flags: Whether to use terminal flags. Default: False.
    :param use_encoder: Whether to use an encoder. Default: False.
    :param distQ: Whether to use distributional Q-learning. Default: False.
    :param n_frames: The number of frames (primarily useful for image inputs). Default: 1.
    :param n_steps: The number of look-ahead steps for N-step temporal
        difference learning. Default: 1.
    :param n_critics: The number of critics. Default: 1.
    :param hidden_units: List of integers denoting the sizes of the hidden layers. Default: [512, 512, 256, 256].
    :param train_eval: Whether to use the training dataset instead of the
        validation dataset during evaluation. Default: False.
    :param use_batch_norm: Whether to use batch normalization. Default: True.
    :param use_dropout: Whether to use dropout. Default: True.
    :param use_dense: Whether to use dense layers. Default: True.
    :param tensorboard_dir: The tensorboard directory. Default: "tensorboard_log".
    :param fit_evaluator: Whether to fit the Fitted Q Evaluation (FQE) model. Default: True.
    :return: None.
    """
    # Loading Data CSV files.
    train_set, val_set, test_set = read_data_csvs()

    # Checking GPU Availability
    use_gpu = torch.cuda.is_available()
    print(f"CHECK -> Using GPU: {use_gpu}")

    # Printing dataset prefix as a sanity check.
    print(f"Dataset prefix: {dataset_prefix}")

    # Initialize algorithm using the specified algorithm label.
    algo = initialize_algo(algo_label)

    # Generate experiment name
    experiment_name = generate_experiment_name(
        use_clusters,
        discrete_action,
        apache2_rewards,
        terminal_flags,
        use_encoder,
        distQ,
        algo_label,
        n_epochs,
        n_steps,
        n_frames,
        n_critics,
        hidden_units,
        train_eval,
        use_batch_norm,
        use_dropout,
        use_dense,
        experiment_label=exp_label
    )
    print(f"RUNNING EXPERIMENT: {experiment_name}...")

    # Load d3rlpy datasets
    train_dataset, val_dataset, test_dataset = load_data(
        train_set, val_set, test_set,
        use_clusters, discrete_action, apache2_rewards, terminal_flags,
        path=f"../data_processing/data/{dataset_prefix}{sampling_window_hours}hour_window/generated_datasets/"
    )

    # Create scalers and encoder
    action_scaler, reward_scaler = load_scalers(discrete_action)
    if use_encoder:
        encoder = VectorEncoderFactory(
            hidden_units=hidden_units,
            use_dense=use_dense,
            use_batch_norm=use_batch_norm,
            dropout_rate=0.2 if use_dropout else None
        )
    else:
        encoder = 'default'

    # Initialize Model. Although model-free algorithms generally require the same
    # hyperparameters/arguments, we have to initialize the model-based methods
    # independently due to the dependence on algorithm-specific arguments.
    if algo_label != "COMBO" and algo_label != "DiscreteCOMBO" and algo_label != "DiscreteCOUPLe":
        model = algo(
            encoder_factory=encoder,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            n_steps = n_steps,
            n_frames = n_frames,
            n_critics = n_critics,
            q_func_factory='qr' if distQ else 'mean',
            use_gpu=use_gpu
        )
    elif algo_label == "DiscreteCOMBO":
        if custom_dynamics:
            dynamics_model = d3rlpy.dynamics.CustomProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)
        else:
            dynamics_model = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)

        print(f"-> Training DiscreteCOMBO with alpha/beta: {combo_alpha} & real_ratio: {real_ratio}...\n")
        model = algo(
            encoder_factory=encoder,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            n_steps = n_steps,
            n_frames = n_frames,
            n_critics = n_critics,
            alpha=combo_alpha,
            q_func_factory='qr' if distQ else 'mean',
            rollout_interval=rollout_interval,
            rollout_horizon=rollout_horizon,
            rollout_batch_size=rollout_batch_size,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
            dynamics=dynamics_model,
            use_gpu=use_gpu
        )
    elif algo_label == "DiscreteCOUPLe":
        if custom_dynamics:
            dynamics_model = d3rlpy.dynamics.CustomProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)
        else:
            dynamics_model = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)

        print(f"-> Training DiscreteCOUPLe with alpha/beta: {combo_alpha} & Lambda: {couple_lam} & real_ratio: {real_ratio}...\n")
        model = algo(
            encoder_factory=encoder,
            reward_scaler=reward_scaler,
            n_steps = n_steps,
            n_frames = n_frames,
            n_critics = n_critics,
            alpha=combo_alpha,
            lam=couple_lam,
            q_func_factory='qr' if distQ else 'mean',
            rollout_interval=rollout_interval,
            rollout_horizon=rollout_horizon,
            rollout_batch_size=rollout_batch_size,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
            dynamics=dynamics_model,
            use_gpu=use_gpu
        )
    else:
        dynamics_model = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
        dynamics_model.load_model(dynamics_checkpoint)
        model = algo(
            actor_encoder_factory=encoder,
            critic_encoder_factory=encoder,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            n_steps = n_steps,
            n_frames = n_frames,
            n_critics = n_critics,
            q_func_factory='qr' if distQ else 'mean',
            rollout_interval=rollout_interval,
            rollout_horizon=rollout_horizon,
            rollout_batch_size=rollout_batch_size,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
            dynamics=dynamics_model,
            use_gpu=use_gpu
        )

    if algo_label == "Dynamics":
        # Dynamics models are evaluated using Observation Error, Reward Error, and Variance.
        scorers = {
            'observation_error': dynamics_observation_prediction_error_scorer,
            'reward_error': dynamics_reward_prediction_error_scorer,
            'variance': dynamics_prediction_variance_scorer,
        }
    else:
        # Standard algorithms are evaluated using TD-Error, Initial State Value, and Average State Value.
        scorers = {
            'td_error': td_error_scorer,
            'initial_state_value': initial_state_value_estimation_scorer,
            'avg_state_value': average_value_estimation_scorer
        }

    # Fit the model. Note: we use n_steps_per_epoch and n_steps instead of n_epochs
    # to ensure that the model is trained for a fixed number of steps. This makes
    # it easier to automatically load the final model after training.
    model.fit(
        train_dataset.episodes,
        eval_episodes=train_dataset.episodes if train_eval else val_dataset.episodes,
        # n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_steps=(n_steps_per_epoch * n_epochs),
        scorers=scorers,
        tensorboard_dir=f"results/{tensorboard_dir}",
        experiment_name=experiment_name,
        with_timestamp=False
    )

    # If fit_evaluator is False or if the model is a dynamics model, we do not
    # evaluate the model.
    if not fit_evaluator or algo_label == "Dynamics":
        return

    # Fit a Fitted Q Evaluation (FQE) model using the trained model.
    evaluate_model(
        algo_label,
        train_dataset,
        test_dataset,
        f"./d3rlpy_logs/{experiment_name}/model_{n_epochs_eval * n_steps_per_epoch_eval}.pt",
        experiment_name,
        discrete_action,
        encoder,
        action_scaler,
        reward_scaler,
        n_steps,
        n_frames,
        n_critics,
        distQ
    )


def train_models():
    """
    train_models: Train all models using the settings specified in training_config.py.

    :return: None
    """
    setting_combinations = load_settings()

    for (use_clusters, discrete_action, apache2_rewards, terminal_flags, \
         use_encoder, distQ, train_eval, use_batch_norm, use_dropout, use_dense, \
         hidden_units, n_frames, n_steps, n_critics) in tqdm(setting_combinations):

        train_model(
            algo_label=algo_label,
            use_clusters=use_clusters,
            discrete_action=discrete_action,
            apache2_rewards=apache2_rewards,
            terminal_flags=terminal_flags,
            use_encoder=use_encoder,
            distQ=distQ,
            n_frames=n_frames,
            n_steps=n_steps,
            n_critics=n_critics,
            hidden_units=hidden_units,
            train_eval=train_eval,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            use_dense=use_dense,
            tensorboard_dir=tensorboard_dir
        )


if __name__ == '__main__':
    set_seed(SEED)
    train_models()