"""
evaluation.py: Provides functions for the evaluation of trained models.
"""
import os
import io
import sys
import numpy as np
import itertools
import torch
import json

import d3rlpy
from d3rlpy.ope import *
from d3rlpy.algos import *

sys.path.insert(0, '../data_processing/scripts/')
from utils import *
from config import *

from training_config import *
from evaluation_config import *
from train_models import *

from evaluator import PolicyEvaluator, PhysicianEvaluator, MaxEvaluator

from tqdm import tqdm


def evaluate_model(
        algo_label,
        train_data,
        test_data,
        policy_savepath,
        experiment_name,
        discrete_action,
        encoder,
        action_scaler,
        reward_scaler,
        n_steps,
        n_frames,
        n_critics,
        distQ
    ):
    """
    evaluate_model: Evaluates a trained model on the test dataset.

    :param algo_label: The algorithm to be evaluated.
    :param train_data: The training dataset.
    :param test_data: The test dataset.
    :param policy_savepath: The path to the trained policy that is to be evaluated.
    :param experiment_name: The name of the experiment.
    :param discrete_action: Whether the action space is discrete.
    :param encoder: The encoder to use for the algorithm.
    :param action_scaler: The action scaler to use for the algorithm.
    :param reward_scaler: The reward scaler to use for the algorithm.
    :param n_steps: The number of look-ahead steps for N-step temporal
        difference learning.
    :param n_frames: The number of frames (primarily useful for image inputs).
    :param n_critics: The number of critics to use for the algorithm.
    :param distQ: Whether to use a distributional Q function.
    :return: The trained FQE evaluator model.
    """
    # Checking GPU Availability
    use_gpu = torch.cuda.is_available()

    # Initialize the algorithm to be evaluated:
    algo = initialize_algo(algo_label)

    if algo_label == "DiscreteCOUPLe":
        if custom_dynamics:
            dynamics_model = d3rlpy.dynamics.CustomProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)
        else:
            dynamics_model = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(f"{dynamics_path}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)

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

    # Initialize the policy with the training dataset.
    model.build_with_dataset(train_data)

    # Loading the trained policy from the specified savepath.
    model.load_model(policy_savepath)

    # Initialize evaluator:
    if discrete_action:
        # When using discrete actions, we use the DiscreteFQE evaluator.
        evaluator = DiscreteFQE(
            algo=model,
            gamma=GAMMA,
            encoder_factory=encoder,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            n_steps = n_steps,
            n_frames = n_frames,
            n_critics = 1,
            q_func_factory='qr' if distQ else 'mean',
            use_gpu=use_gpu
        )
    else:
        # When using continuous actions, we use the FQE evaluator.
        evaluator = FQE(
            algo=model,
            gamma=GAMMA,
            encoder_factory=encoder,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            n_steps = n_steps,
            n_frames = n_frames,
            n_critics = 1,
            q_func_factory='qr' if distQ else 'mean',
            use_gpu=use_gpu
        )

    # Fit evaluator:
    evaluator.fit(
        train_data,
        eval_episodes=test_data,
        n_steps_per_epoch=n_steps_per_epoch_eval,
        n_steps=(n_steps_per_epoch_eval * n_epochs_eval),
        scorers={
            'td_error': td_error_scorer,
            'initial_state_value': initial_state_value_estimation_scorer,
            'avg_state_value': average_value_estimation_scorer
        },
        tensorboard_dir=f"results/FQE/{tensorboard_dir}",
        experiment_name=f"FQE_{experiment_name}",
        with_timestamp=False
    )

    return evaluator


def load_datasets(experiment_name):
    """
    load_datasets: Loads the datasets for the specified experiment. This function
        extracts the appropriate dataset configurations from the labels in the
        experiment name.

    :param experiment_name: The name of the experiment.
    :return: The training, validation and test datasets.
    """
    print("Loading datasets...")
    discrete_action = "DiscA" in experiment_name
    use_clusters = "Clusters" in experiment_name
    apache2_rewards = "Apache2" in experiment_name
    terminal_flags = "TermFlags" in experiment_name
    print(f"For {experiment_name}, the following configuration was extracted:")
    print(f" - Discrete_action = {discrete_action}")
    print(f" - Clusters = {use_clusters}")
    print(f" - Apache2 intermediate rewards = {apache2_rewards}")
    print(f" - Terminal Flags = {terminal_flags}")

    # Load datasets
    train_set, val_set, test_set = read_data_csvs()
    train_dataset, val_dataset, test_dataset = load_data(
        train_set, val_set, test_set,
        use_clusters=use_clusters,
        discrete_action=discrete_action,
        apache2_rewards=apache2_rewards,
        terminal_flags=terminal_flags,
        path=f"../data_processing/data/{sampling_window_hours}hour_window/generated_datasets/"
    )
    print("Datasets loaded successfully!\n")

    return train_dataset, val_dataset, test_dataset


def extract_encoder_params(experiment_name, params):
    """
    extract_encoder_params: Extracts the encoder parameters from the experiment
        name and the params file.

    :param experiment_name: The name of the experiment.
    :param params: The params file of the trained policy.
    :return: The encoder.
    """
    if "Encoded" in experiment_name:
        encoder_params = params["encoder_factory"]["params"]
        encoder = VectorEncoderFactory(
            hidden_units=encoder_params["hidden_units"],
            use_dense=encoder_params["use_dense"],
            use_batch_norm=encoder_params["use_batch_norm"],
            dropout_rate=0.2 if encoder_params["dropout_rate"] != None else None
        )
    else:
        encoder = 'default'

    return encoder


def load_config(log_path, experiment_name):
    """
    load_config: Loads the configuration of the specified experiment.

    :param log_path: The path to the log directory.
    :param experiment_name: The name of the experiment.
    :return: The configuration dictionary for the trained model.
    """
    # Open params file:
    f = open(f"./{log_path}/{experiment_name}/params.json")
    params = json.load(f)

    # Extract encoder:
    encoder = extract_encoder_params(experiment_name, params)

    # Derive experiment configuration:
    discrete_action = "DiscA" in experiment_name
    action_scaler, reward_scaler = load_scalers(discrete_action)
    n_steps = params["n_steps"]
    n_frames = params["n_frames"]
    n_critics = params["n_critics"]
    distQ = "distQ" in experiment_name

    model_based = ("COUPLe" in experiment_name) or ("COMBO" in experiment_name)
    real_ratio = params["real_ratio"] if model_based else None
    alpha = params["alpha"] if model_based else None
    lam = params["lam"] if model_based else None
    rollout_batch_size = params["rollout_batch_size"] if model_based else None
    rollout_horizon = params["rollout_horizon"] if model_based else None
    rollout_interval = params["rollout_interval"] if model_based else None
    generated_maxlen = params["generated_maxlen"] if model_based else None

    # Store parameters in configuration dictionary:
    model_config = {
        "encoder": encoder,
        "discrete_action": discrete_action,
        "action_scaler": action_scaler,
        "reward_scaler": reward_scaler,
        "n_steps": n_steps,
        "n_frames": n_frames,
        "n_critics": n_critics,
        "distQ": distQ,
        "model_based": model_based,
        "real_ratio": real_ratio,
        "alpha": alpha,
        "lam": lam,
        "rollout_batch_size": rollout_batch_size,
        "rollout_horizon": rollout_horizon,
        "rollout_interval": rollout_interval,
        "generated_maxlen": generated_maxlen
    }

    return model_config


def evaluate_policies(log_path="d3rlpy_logs"):
    """
    evaluate_policies: Evaluates the policies of the specified experiments.

    :param log_path: The path to the log directory.
    :return: None.
    """
    # Checking GPU Availability
    use_gpu = torch.cuda.is_available()

    train_results, test_results = {}, {}
    for (algo_label, experiment_label, experiment_name, checkpoint) in tqdm(TO_EVALUATE):
        policy_savepath = f"./{log_path}/{experiment_name}/{checkpoint}.pt"
        eval_savepath = f"./{log_path}/FQE_{experiment_name}/{checkpoint}.pt"

        # Load datasets:
        train_dataset, val_dataset, test_dataset = load_datasets(experiment_name)

        # Load model configuration:
        model_config = load_config(log_path, experiment_name)
        discrete_action = "DiscA" in experiment_name

        # Initialize the algorithm to be evaluated:
        algo = initialize_algo(algo_label)

        # Initialize the FQE evaluator:
        evaluator = PolicyEvaluator(
            f"{algo_label}{experiment_label}",
            train_dataset, val_dataset, test_dataset,
            discrete_action, algo, policy_savepath,
            eval_savepath, model_config,
            dynamics_savepath=dynamics_path_eval,
            dynamics_checkpoint=dynamics_checkpoint_eval
        )

        # Evaluate the policy:
        train_values, train_stds = evaluator.estimate_initial_state_values(evaluator.train_data)
        train_results[f"{algo_label}{experiment_label}"] = train_values.mean()

        test_values, test_stds = evaluator.estimate_initial_state_values(evaluator.test_data)
        test_results[f"{algo_label}{experiment_label}"] = test_values.mean()

    # Evaluate the physician and maximum policies:
    physician = PhysicianEvaluator(train_dataset, val_dataset, test_dataset, discrete_action=True)
    phys_train_values = physician.estimate_initial_state_values(physician.train_data)
    phys_test_values = physician.estimate_initial_state_values(physician.test_data)
    train_results["Physician"] = phys_train_values.mean()
    test_results["Physician"] = phys_test_values.mean()

    max_evaluator = MaxEvaluator(train_dataset, val_dataset, test_dataset, discrete_action=True)
    max_train_values = max_evaluator.estimate_initial_state_values(max_evaluator.train_data)
    max_test_values = max_evaluator.estimate_initial_state_values(max_evaluator.test_data)
    train_results["Maximum"] = max_train_values.mean()
    test_results["Maximum"] = max_test_values.mean()

    # Print results:
    print(f"Training set results: {train_results}")
    print()
    print(f"Test set results: {test_results}")
    print()


if __name__ == "__main__":
    evaluate_policies()
