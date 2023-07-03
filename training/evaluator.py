"""
evaluator.py: Contains the evaluator classes for evaluating trained policies.
Inspired by: https://github.com/FlemmingKondrup/DeepVent/blob/main/evaluation/estimator.py
"""
import os
import io
import sys
import numpy as np
import itertools
import torch

import d3rlpy
from d3rlpy.ope import *
from d3rlpy.algos import *

sys.path.insert(0, '../data_processing/scripts/')
from utils import *
from config import *

from training_config import *
from evaluation_config import *

class Evaluator():
    """
    Evaluator class for evaluating agents.
    """
    def __init__(self, label, train_data, val_data, test_data, discrete_action, model_config):
        self.label = label
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.discrete_action = discrete_action
        self.model_config = model_config

class PolicyEvaluator(Evaluator):
    """
    PolicyEvaluator class for evaluating trained policies.
    """
    def __init__(self, label, train_data, val_data, test_data, discrete_action, algo, model_savepath, eval_savepath, model_config, dynamics_savepath=None, dynamics_checkpoint=None):
        """
        __init__

        :param label: The label of the policy.
        :param train_data: The training dataset.
        :param val_data: The validation dataset.
        :param test_data: The test dataset.
        :param discrete_action: Whether the action space is discrete.
        :param algo: The algorithm used to train the policy.
        :param model_savepath: The savepath of the trained policy.
        :param eval_savepath: The savepath of the trained FQE evaluation model.
        :param model_config: Dictionary of the trained model configuration.
            Necessary to initialize the models for evaluation.
        :param dynamics_savepath: The savepath of the dynamics model. Default: None.
        :param dynamics_checkpoint: The checkpoint of the dynamics model. Default: None.
        :return: None.
        """
        super().__init__(label, train_data, val_data, test_data, discrete_action, model_config)
        self.algo = algo
        self.model_savepath = model_savepath
        self.eval_savepath = eval_savepath
        self.dynamics_savepath = dynamics_savepath
        self.dynamics_checkpoint = dynamics_checkpoint
        self.dynamics_model = None

        # If dynamics savepath and checkpoint are specified, load the dynamics model.
        if (self.dynamics_savepath is not None) and (self.dynamics_checkpoint is not None):
            self.dynamics_model = self.load_dynamics_model(dynamics_savepath, dynamics_checkpoint)
        self.policy = self.load_policy(model_savepath)
        self.FQE = self.load_FQE(eval_savepath)

    def load_dynamics_model(self, dynamics_savepath, dynamics_checkpoint, use_custom_dynamics=True):
        """
        load_dynamics_model: Loads the dynamics model from the specified savepath and checkpoint.

        :param dynamics_savepath: The savepath of the dynamics model.
        :param dynamics_checkpoint: The checkpoint of the dynamics model.
        :param use_custom_dynamics: Whether to use the custom dynamics model or
            the default dynamics model.
        :return: The loaded dynamics model.
        """
        if use_custom_dynamics:
            dynamics_model = d3rlpy.dynamics.CustomProbabilisticEnsembleDynamics.from_json(f"{dynamics_savepath}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)
        else:
            dynamics_model = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json(f"{dynamics_savepath}/params.json")
            dynamics_model.load_model(dynamics_checkpoint)

        return dynamics_model

    def load_policy(self, model_savepath):
        """
        load_policy: Loads the policy from the specified savepath.

        :param model_savepath: The savepath of the policy.
        :return: The loaded policy.
        """
        # Model-based methods contain more hyperparameters and are therefore
        # handled separately from model-free methods.
        if self.model_config["model_based"]:
            policy = self.algo(
                encoder_factory=self.model_config["encoder"],
                action_scaler = self.model_config["action_scaler"],
                reward_scaler = self.model_config["reward_scaler"],
                n_steps = self.model_config["n_steps"],
                n_frames = self.model_config["n_frames"],
                n_critics = self.model_config["n_critics"],
                alpha=self.model_config["alpha"],
                lam=self.model_config["lam"],
                q_func_factory='qr' if self.model_config["distQ"] else 'mean',
                rollout_interval=self.model_config["rollout_interval"],
                rollout_horizon=self.model_config["rollout_horizon"],
                rollout_batch_size=self.model_config["rollout_batch_size"],
                real_ratio=self.model_config["real_ratio"],
                generated_maxlen=self.model_config["generated_maxlen"],
                dynamics=self.dynamics_model
        )
        else:
            policy = self.algo(
                encoder_factory = self.model_config["encoder"],
                action_scaler = self.model_config["action_scaler"],
                reward_scaler = self.model_config["reward_scaler"],
                n_steps = self.model_config["n_steps"],
                n_frames = self.model_config["n_frames"],
                n_critics = self.model_config["n_critics"],
                q_func_factory='qr' if self.model_config["distQ"] else 'mean'
            )

        # Initialise the policy using the training dataset.
        policy.build_with_dataset(self.train_data)

        # Load the policy from the specified savepath.
        policy.load_model(model_savepath)
        return policy

    def load_FQE(self, eval_savepath):
        """
        load_FQE: Loads the FQE model from the specified savepath.

        :param eval_savepath: The savepath of the trained FQE model.
        :return: The loaded FQE model.
        """

        if self.discrete_action:
            # If the action space is discrete, use the DiscreteFQE algorithm.
            evaluator = DiscreteFQE(
                algo=self.policy,
                gamma=GAMMA,
                encoder_factory = self.model_config["encoder"],
                action_scaler = self.model_config["action_scaler"],
                reward_scaler = self.model_config["reward_scaler"],
                n_steps = 1,
                n_frames = 1,
                n_critics = 1,
                q_func_factory='qr' if self.model_config["distQ"] else 'mean'
            )
        else:
            # If the action space is continuous, use the FQE algorithm.
            evaluator = FQE(
                algo=self.policy,
                gamma=GAMMA,
                encoder_factory = self.model_config["encoder"],
                action_scaler = self.model_config["action_scaler"],
                reward_scaler = self.model_config["reward_scaler"],
                n_steps = 1,
                n_frames = 1,
                n_critics = 1,
                q_func_factory='qr' if self.model_config["distQ"] else 'mean'
            )

        # Initialise the FQE model using the training dataset.
        evaluator.build_with_dataset(self.train_data)

        # Load the FQE model from the specified savepath.
        evaluator.load_model(eval_savepath)
        return evaluator

    def estimate_initial_state_values(self, dataset):
        """
        estimate_initial_state_values: Estimates the initial state values of
            the dataset using the FQE model.

        :param dataset: The dataset for which to estimate the initial state values.
        :return: The estimated initial state values.
        """
        initial_states = np.array([episode.observations[0] for episode in dataset.episodes])

        # For our experiments, we assume that the policy will always take the
        # greedy action during inference.
        greedy_actions = self.FQE.predict(initial_states)
        return self.FQE.predict_value(initial_states, greedy_actions, with_std=True)


class PhysicianEvaluator(Evaluator):
    """
    PhysicianEvaluator: An evaluator for the physician policy.
    """
    def __init__(self, train_data, val_data, test_data, discrete_action):
        """
        __init__

        :param train_data: The training dataset.
        :param val_data: The validation dataset.
        :param test_data: The test dataset.
        :param discrete_action: Whether the action space is discrete.
        :return: None.
        """
        super().__init__("Physician", train_data, val_data, test_data, discrete_action, model_config=None)

    def estimate_initial_state_values(self, dataset):
        """
        estimate_initial_state_values: Estimates the initial state values of
            the dataset using the physician policy.

        :param dataset: The dataset for which to estimate the initial state values.
        :return: The estimated initial state values.
        """
        # Derive offset: For some dataset variants there is an additional
        # "terminal state", while for others the episode terminates with the
        # last measured timestep of the ICU stay. In the former case, the
        # episode reward is given in the second-to-last timestep, and in the
        # latter the reward is stored in the last timestep:
        offset = 0 if dataset.episodes[0].rewards[-1] != 0 else 1

        # For the evaluation of the physician policy, we directly take the
        # observed returns in the dataset as the initial state values.
        initial_state_values = []
        for episode in dataset.episodes:
            reward = episode.rewards[-1 - offset]

            discount = GAMMA ** (len(episode) - offset)
            initial_state_values.append(reward * discount)

        return np.array(initial_state_values)

class MaxEvaluator(Evaluator):
    """
    MaxEvaluator: An evaluator for the maximum obtainable return.
    """
    def __init__(self, train_data, val_data, test_data, discrete_action):
        """
        __init__

        :param train_data: The training dataset.
        :param val_data: The validation dataset.
        :param test_data: The test dataset.
        :param discrete_action: Whether the action space is discrete.
        :return: None.
        """
        super().__init__("Maximum", train_data, val_data, test_data, discrete_action, model_config=None)

    def estimate_initial_state_values(self, dataset):
        """
        estimate_initial_state_values: Estimates the average maximum obtainable
            return in the specified dataset. To obtain the maximum value, we
            derive the average return if every patient in the dataset would have
            survived, and assume that the episode lengths in the dataset do not
            change (as this would affect the returns due to the discount factor).

        :param dataset: The dataset for which to estimate the initial state values.
        :return: The estimated initial state values.
        """
        # Derive offset: For some dataset variants there is an additional
        # "terminal state", while for others the episode terminates with the
        # last measured timestep of the ICU stay. In the former case, the
        # episode reward is given in the second-to-last timestep, and in the
        # latter the reward is stored in the last timestep:
        offset = 0 if dataset.episodes[0].rewards[-1] != 0 else 1

        initial_state_values = []
        for episode in dataset.episodes:
            discount = GAMMA ** (len(episode) - offset)
            initial_state_values.append(REWARD * discount)

        return np.array(initial_state_values)
