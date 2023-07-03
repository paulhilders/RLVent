"""
integrated_gradients.py: Contains functions for computing Integrated Gradients (IG)
    attribution maps for a given model.
"""
import os
import io
import sys
import numpy as np
import itertools
import torch
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import d3rlpy

sys.path.insert(0, '../data_processing/scripts/')
from utils import *
from config import *

from evaluation import *

def discretize_action(vt, peep, fio2, levels=7):
    """
    discretize_action: Discretizes the action space into a single integer.

    :param vt: Discrete level of Tidal volume
    :param peep: Discrete level of Positive end-expiratory pressure
    :param fio2: Discrete level of Fraction of inspired oxygen
    :param levels: Number of discrete levels for each action
    :return: Discrete action
    """
    return vt * (levels) ** 2 + peep * (levels) + fio2

def separate_action(action_int, levels=7):
    """
    separate_action: Separates the discrete action into its components.

    :param action_int: Discrete action
    :param levels: Number of discrete levels for each action
    :return: Tidal volume, PEEP, FiO2
    """
    # Separate FiO2
    fio2 = action_int % levels

    # Separate PEEP
    peep = ((action_int - fio2) / levels) % levels

    # Separate Vt
    vt = ((action_int - fio2) - levels * peep) / (levels**2)

    return vt, peep, fio2


def generate_action_masks(n_actions=343, levels=7):
    """
    generate_action_masks: Generates action masks for the different actions.
        These masks can be used to separate the actions in the dataset into
        more abstract categories. Specifically, we use the masks to separate
        each action dimension into a low-level and high-level action.

    :param n_actions: Number of actions
    :param levels: Number of discrete levels for each action
    :return: Dictionary of action masks
    """
    masks = {
        "vt_low": np.zeros(n_actions),
        "vt_high": np.zeros(n_actions),
        "peep_low": np.zeros(n_actions),
        "peep_high": np.zeros(n_actions),
        "fio2_low": np.zeros(n_actions),
        "fio2_high": np.zeros(n_actions)
    }
    vt_low, vt_high, peep_low, peep_high, fio2_low, fio2_high = [], [], [], [], [], []

    for action_int in range(n_actions):
        vt, peep, fio2 = separate_action(action_int)

        # For each action, we define a low-level and high-level action.
        # "Neutral" actions, i.e. actions with the middle level of 4, are
        # ignored.
        if vt < 3:
            vt_low.append(action_int)
        if vt > 3:
            vt_high.append(action_int)

        if peep < 3:
            peep_low.append(action_int)
        if peep > 3:
            peep_high.append(action_int)

        if fio2 < 3:
            fio2_low.append(action_int)
        if fio2 > 3:
            fio2_high.append(action_int)

    masks["vt_low"][vt_low] = 1
    masks["vt_high"][vt_high] = 1
    masks["peep_low"][peep_low] = 1
    masks["peep_high"][peep_high] = 1
    masks["fio2_low"][fio2_low] = 1
    masks["fio2_high"][fio2_high] = 1

    return masks


class IntegratedGradients:
    """
    IntegratedGradients: Class for computing Integrated Gradients (IG)
        attribution maps for a given model.

    Inspired by: https://github.com/thomas097/Haemodynamic-Optimization-Reinforcement-Learning/blob/main/experiments/plotting/integrated_gradients.py
    """
    def __init__(self, n_interpolations, baseline, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        __init__

        :param n_interpolations: Number of interpolations between the input
            and baseline.
        :param baseline: Baseline input for the IG algorithm.
        :param device: Device to run the IG algorithm on.
        """
        self.n_interpolations = n_interpolations
        self.baseline = baseline
        self.device = device

    def _linear_interpolation(self, input, baseline):
        """
        _linear_interpolation: Performs linear interpolation between the
            input and baseline.

        :param input: Input to the model.
        :param baseline: Baseline input for the IG algorithm.
        :return: Interpolated input.
        """
        input = input.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        interpolations = torch.linspace(0, 1, self.n_interpolations).unsqueeze(1).unsqueeze(1).to(self.device)
        delta = input - baseline
        return baseline + interpolations * delta

    def __call__(self, input, model, mask=None):
        """
        __call__: Computes the IG attribution map for the given input and model.

        :param input: Input to the model.
        :param model: Model to compute the IG attribution map for.
        :param mask: Mask to apply to the IG attribution map.
        :return: IG attribution map.
        """
        input = input.to(self.device)
        baseline = self.baseline.repeat((input.shape[0], 1)).to(self.device)
        model = model.to(self.device)

        interpolated_input = self._linear_interpolation(input, baseline)

        gradients = []
        for sample in tqdm(interpolated_input):
            sample_copy = sample.detach().clone()
            sample_copy.requires_grad = True

            model.zero_grad()

            # Compute the gradients:
            if mask is None:
                output = model(sample_copy)[0]
                action_index = torch.argmax(output)
                target = output[action_index]
                target.backward()
            else:
                output = model(sample_copy)
                target = torch.mul(output, torch.from_numpy(mask).to(self.device))
                target = torch.mean(target)
                target.backward()

            gradient = sample_copy.grad.cpu().detach().clone()
            gradients.append(gradient)
        gradients = torch.concat(gradients, dim=0)

        # Riemann approximation of area under the gradients:
        avg_gradients = torch.mean((gradients[:-1] + gradients[1:]) / 2, dim=0).to(self.device)

        # Scale the gradients to obtain the attribution map:
        final_map = avg_gradients * (input - baseline)

        return torch.mean(final_map, dim=0).cpu().detach().numpy()


def visualize_IG_attributions(attribution_maps, save_name=None):
    """
    visualize_IG_attributions: Visualizes the IG attribution maps.

    :param attribution_maps: Dictionary of IG attribution maps.
    :param save_name: Name base of the file to save the figures to.
    :return: None.
    """
    state_variables, state_varnames = get_state_varnames()
    features = state_variables + ['Terminal State', 'Expired State']

    sns.set(style="white")
    cmap = sns.color_palette("RdBu", 32)
    f, ax = plt.subplots(figsize=(12,12))

    # Convert action map values to numpy array and transpose.
    values = np.array(list(attribution_maps.values())).T

    # Plot heatmap:
    ax = sns.heatmap(
        values,
        cmap=cmap,
        xticklabels=[str(key) for key in attribution_maps.keys()],
        yticklabels=features,
        linewidths=0.5,
        robust=True,
        center=0.0,
        annot=True
    )
    ax.xaxis.tick_top()

    # Plot or save figure:
    if save_name is None:
        plt.show(block=True)
    else:
        f.savefig(f"ig_results/{save_name}.jpg", bbox_inches='tight')
        f.savefig(f"ig_results/{save_name}.pdf", bbox_inches='tight')


def apply_integrated_gradients():
    """
    apply_integrated_gradients: Applies the Integrated Gradients algorithm to
        the trained model.

    :return: None.
    """
    # Checking GPU Availability
    use_gpu = torch.cuda.is_available()
    print(f"CHECK -> Using GPU: {use_gpu}\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load trained model
    checkpoint_path = "d3rlpy_logs/DiscreteCQL_FinalRun_20epochs_DiscA_TermFlags_Encoded[1024,1024,512,256,256](BN)(Dense)_ValEval"
    checkpoint = "model_2000000.pt"
    model = d3rlpy.algos.DiscreteCQL.from_json(f'{checkpoint_path}/params.json')
    model.load_model(f'{checkpoint_path}/{checkpoint}')

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(experiment_name=checkpoint_path)

    # Set baseline to average value from the training dataset
    baseline = torch.mean(torch.from_numpy(train_dataset.observations), dim=0)

    # Generate masks for different action categories:
    masks = generate_action_masks()

    n_interpolations = 64
    att_maps = {}
    for key, mask in masks.items():
        IG = IntegratedGradients(n_interpolations=n_interpolations, baseline=baseline, device=device)
        samples = torch.from_numpy(test_dataset.observations)
        att = IG(samples, model.impl.q_function, mask=mask)
        att_maps[key] = att

    visualize_IG_attributions(att_maps, save_name=f"CQL_attribution_map")


if __name__ == "__main__":
    apply_integrated_gradients()
