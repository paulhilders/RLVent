"""
training_config.py: Contains all the training configurations for the training
    script.
"""
SEED = 42
n_epochs = 20
n_steps_per_epoch = 100000
algo_label = "DiscreteCOUPLe"
exp_label = "" # E.g. "Run2" or "Final"
tensorboard_dir = "tensorboard_log_models"

# Setting training configurations
use_clusters_options =    [False]
discrete_action_options = [True] # E.g. True for DiscreteCQL and False for CQL / Dynamics / COMBO.
apache2_rewards_options = [False]
terminal_flags_options =  [True]
use_encoder_options =     [True] # Default = True
distQ_options =           [False]
train_eval_options =      [False]

# Architecture options
use_batch_norm_options =  [True]
use_dropout_options =     [False]
use_dense_options =       [True]
hidden_units_options = [
    (1024,1024,512,256,256),
    # (2048,2048,1024,1024,512,512,256,256),
]

n_frames_options =        [1]
n_steps_options =         [1]
n_critics_options =       [2]

# Set dynamics model to be used:
custom_dynamics = True
dynamics_path = "./d3rlpy_logs/CustomDynamics_50epochs"
dynamics_checkpoint = f"{dynamics_path}/model_1700000.pt"

# Model-based parameters. (Not used for model-free algorithms)
real_ratio = 0.05
rollout_interval = 10000
rollout_horizon = 1
rollout_batch_size = 20000
generated_maxlen = 300000
combo_alpha = 1.0 # Default is 1.0. Higher alpha increases conservativeness.
couple_lam = 1.0