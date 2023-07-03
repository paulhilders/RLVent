"""
evaluation_config.py: Contains all the evaluation configurations for the
    evaluation script.
"""
n_epochs_eval = 20
n_steps_per_epoch_eval = 100000
n_steps_eval = n_epochs_eval * n_steps_per_epoch_eval

# Dynamics to use for model-based algorithms:
dynamics_path_eval = "./d3rlpy_logs/CustomDynamics_50epochs"
dynamics_checkpoint_eval = f"{dynamics_path_eval}/model_1700000.pt"
custom_dynamics_eval = True

# Evaluation configurations
# Format: (Algorithm, Experiment label, Experiment savename, Checkpoint)
TO_EVALUATE = [
    (
        "DiscreteCQL",
        "_MIMIC",
        "DiscreteCQL_FinalRun_20epochs_DiscA_TermFlags_Encoded[1024,1024,512,256,256](BN)(Dense)_ValEval",
        f"model_{n_steps_eval}"
    ), (
        "DiscreteCOUPLe",
        "_MIMIC",
        "DiscreteCOUPLe_FinalRun_20epochs_DiscA_TermFlags_Encoded[1024,1024,512,256,256](BN)(Dense)_ValEval",
        f"model_{n_steps_eval}"
    )
]

# Include physician performance in evaluation comparison
INCLUDE_PHYSICIAN = True

# Include maximum performance in evaluation comparison
INCLUDE_MAXIMUM = True
