"""
data_processing.py: Provides functions for converting the raw database tables
    into a fully processed CSV file that can be translated into a dataset.
"""
import os
import io
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.impute import KNNImputer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from config import demographical_vars, observational_vars, state_vars, sampling_window_hours, Vt_bins, PEEP_bins, FiO2_bins, dataset_prefix


def retrieve_table(file=f'../data/{dataset_prefix}ventilatedpatients_hourly.csv'):
    """
    retrieve_table: Retrieves the table from the specified file.

    :param file: The file to retrieve the table from.
    :return: The table dataframe.
    """
    df = pd.read_csv(file)
    return df


def retrieve_mv_indices(df):
    """
    retrieve_mv_indices: Retrieves the indices of the mechanical ventilation episodes.

    :param df: The dataframe to retrieve the indices from.
    :return: The unique IDs and the indices of the mechanical ventilation episodes.
    """
    # Sort the dataframe appropriately. In this case that means ordering the rows by unique_id and start_time.
    df = df.sort_values(['unique_id', 'start_time']).reset_index().drop(['index', ], axis=1)

    unique_ids = df.unique_id.unique()

    mechvent_episodes_ends = {}
    mechvent_episodes_indices = []

    for unique_id in tqdm(unique_ids):
        try:
            first = df[(df.unique_id == unique_id) & (df.mechvent == 1.0)].index[0]
        except IndexError:
            continue

        try:
            # Grab index of final IMV event (after 'first' index).
            last_idx = df[(df.unique_id == unique_id) & (df.mechvent == 0.0)].index
            last = int(last_idx.where(last_idx > first).dropna()[0])
        except IndexError:
            last = df[(df.unique_id == unique_id)].index[-1]
        mechvent_episodes_ends[unique_id] = (first, last)
        mechvent_episodes_indices = mechvent_episodes_indices + list(np.arange(first, last))

    return unique_ids, mechvent_episodes_indices


def forward_fill(df):
    """
    forward_fill: Perform forward filling or Sample-and-Hold imputation on
        the dataframe.

    :param df: The dataframe to impute.
    :return: The dataframe after applying Sample-and-Hold imputation.
    """
    for col in observational_vars:
        df[col] = df.groupby('unique_id')[col].ffill()
    return df


def tukeys_fence(series, df_capping_values, col, factor=1.5, includena=True):
    """
    tukeys_fence: Applies Tukey's Fences to the series.
    Inspired by: https://github.com/florisdenhengst/ventai/blob/main/notebooks/Analysis%20and%20Preprocessing.ipynb

    :param series: The series to apply Tukey's Fences to.
    :param df_capping_values: The dataframe containing the capping values used
        in our hierarchical two-step outlier detection method.
    :param col: The column to apply Tukey's Fences to.
    :param factor: The factor to multiply the IQR by. Default: 1.5.
    :param includena: Whether to include NA values in the calculation. Default: True.
    :return: The series after applying Tukey's Fences.
    """
    q1, q3 = series.quantile(.25), series.quantile(.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - factor * iqr, q3 + factor * iqr

    limits = (df_capping_values[df_capping_values['parameter'] == col])
    minval = limits['min'].values[0]
    maxval = limits['max'].values[0]

    # Broaden the range, depending on the handpicked capping values.
    lower_fence = minval if lower_fence > minval else lower_fence
    upper_fence = maxval if upper_fence < maxval else upper_fence

    # Edge case: set lower fence to 0 if the minval capping value was set to 0. The values
    # were only set to 0 for variables that cannot be negative.
    lower_fence = 0 if minval == 0 else lower_fence

    valuefilter = (series >= lower_fence) & (series <= upper_fence)
    if includena:
        return valuefilter | series.isna()
    else:
        return valuefilter


def remove_outliers(df, df_imv):
    """
    remove_outliers: Removes outliers from the dataframe.

    :param df: The dataframe to remove outliers from.
    :param df_imv: The dataframe containing the mechanical ventilation episodes.
    :return: The dataframe after removing outliers.
    """
    # Tukeys Fence can be too strict. For example, the method will restrict GCS scores to 15.0 only,
    # even though any value between 3 and 15 is likely valid. Therefore, we include a set of handpicked
    # "capping values", where any value that falls between the designated range is deemed valid.
    # If a value falls outside of the specified range, Tukeys fence is used as a secondary detection method.
    df_capping_values = pd.read_csv('mimiciv_parameter_capping_values.csv')

    # Adapted from: https://github.com/florisdenhengst/ventai/blob/main/notebooks/Analysis%20and%20Preprocessing.ipynb
    boolean_vars = {'mechvent'}
    tukeys_fences = []
    for col in observational_vars:
        if pd.api.types.is_numeric_dtype(df_imv[col]) and col not in boolean_vars:

            df_imv.loc[:, col + '_in_tukeys_fence'] = tukeys_fence(df_imv[col], df_capping_values, col)
            tukeys_fences += [col + '_in_tukeys_fence',]
            df_imv.loc[~df_imv[col + '_in_tukeys_fence'], col] = np.NaN

    ranges = {
        'weight': (25, 400),
        'admission_age': (18, 150),
    }
    for col in ranges:
        lower, upper = ranges[col]
        df_imv.loc[(df_imv[col] < lower) | (df_imv[col] > upper), col] = np.NaN

    # Remove patients with >50% missing values.
    to_remove = ((df_imv[state_vars].isnull().groupby(df.unique_id).mean().mean(axis=1)) > .5)
    print("Removing {} / {} = {}% patients due to >50% missing values".format(to_remove.sum(), len(df_imv.unique_id.unique()), 100*to_remove.sum()/len(df_imv.unique_id.unique())))
    df_imv_f = df_imv[~df_imv.unique_id.isin(to_remove[to_remove].index)]
    selected_patients = df_imv_f.drop_duplicates(subset='unique_id', keep='first')

    print("Selection of {} timesteps for {} patients / mech vent events".format(df_imv_f.shape[0], selected_patients.shape[0]))
    return df_imv_f, selected_patients


def center_and_scale(df_imv_f):
    """
    center_and_scale: Centers and scales the dataframe.

    :param df_imv_f: The dataframe to center and scale.
    :return: The dataframe after centering and scaling, the variable names of
        the scaled variables, and the scalers used to scale the variables.
    """
    # Inspired by: https://github.com/florisdenhengst/ventai/blob/main/notebooks/Analysis%20and%20Preprocessing.ipynb
    numeric_scaler = StandardScaler

    df_stays = df_imv_f.drop_duplicates(subset='unique_id', keep='first')
    scalers = {}

    scaled_vars = []
    for col in observational_vars:
        if pd.api.types.is_numeric_dtype(df_imv_f[col]):
            scaled_vars.append(col)

    observational_vars_scaler = numeric_scaler()
    scaled = observational_vars_scaler.fit_transform(df_imv_f[scaled_vars])
    scaled_observational_varnames = [var + '_scaled' for var in observational_vars_scaler.get_feature_names_out()]
    df_imv_f[scaled_observational_varnames] = scaled.copy()

    demographical_scaled_vars = []
    for col in demographical_vars:
        if pd.api.types.is_numeric_dtype(df_imv_f[col]):
            scaled_vars.append(col)
            demographical_scaled_vars.append(col)

    demographical_vars_scaler = numeric_scaler()
    demographical_vars_scaler.fit(df_stays[demographical_scaled_vars])
    scaled_demographical_varnames = [var + '_scaled' for var in demographical_vars_scaler.get_feature_names_out()]
    scaled = demographical_vars_scaler.transform(df_imv_f[demographical_scaled_vars])
    df_imv_f[scaled_demographical_varnames] = scaled.copy()

    return (
        df_imv_f,
        scaled_vars,
        scaled_observational_varnames,
        demographical_scaled_vars,
        scaled_demographical_varnames,
        observational_vars_scaler,
        demographical_vars_scaler
    )


def knn_imputation(df_imv_f, scaled_varnames, scaled_vars, save_location=f"../data_processors/{dataset_prefix}KNNimputer.bin"):
    """
    knn_imputation: Imputes missing values using KNNImputer.

    :param df_imv_f: The dataframe to impute.
    :param scaled_varnames: The variable names of the scaled variables.
    :param scaled_vars: The variable names of the scaled variables.
    :param save_location: The location to save the imputer.
    :return: The dataframe after imputation, the imputed variables, the imputer,
        and the imputed variable names.
    """
    df_final = df_imv_f.copy()
    imputed_varnames = list(map(lambda x: x + '_final', scaled_vars))
    print(scaled_vars)
    print(scaled_varnames)
    print(imputed_varnames)

    print("Initializing KNNImputer...")
    imputer = KNNImputer(
        n_neighbors=3,
        keep_empty_features=True
    )

    print("Starting KNN imputation...")
    imputed_vars = None
    imputed_vars = imputer.fit_transform(df_final[scaled_varnames].to_numpy())
    df_final.loc[:, imputed_varnames] = imputed_vars

    print("Saving imputer")
    joblib.dump(imputer, save_location, compress=True)

    return df_final, imputed_vars, imputer, imputed_varnames


def dataset_splits(df_imv_f):
    """
    dataset_splits: Splits the dataset into training, validation, and test sets.

    :param df_imv_f: The dataframe to split.
    :return: The training, validation, and test dataframes.
    """
    # Divide into Training / Validation / Test splits using 60%/20%/20%
    train_size, test_size, val_size = .6, .2, .2

    ids = df_imv_f.unique_id.unique()
    print(f"Total number of patients: {ids.shape[0]}")
    remainder, validation_ids = train_test_split(ids, test_size=val_size)
    test_size_cor = test_size / (test_size + train_size) # Compensate for reduced number of ids.
    train_ids, test_ids = train_test_split(remainder, test_size=test_size_cor)

    print(f"Number of training patients: {train_ids.shape[0]} ({100*train_ids.shape[0]/ids.shape[0]}%)")
    print(f"Number of validation patients: {validation_ids.shape[0]} ({100*validation_ids.shape[0]/ids.shape[0]}%)")
    print(f"Number of test patients: {test_ids.shape[0]} ({100*test_ids.shape[0]/ids.shape[0]}%)")

    train_data = df_imv_f[df_imv_f.unique_id.isin(train_ids)]
    val_data = df_imv_f[df_imv_f.unique_id.isin(validation_ids)]
    test_data = df_imv_f[df_imv_f.unique_id.isin(test_ids)]

    total_rows = len(df_imv_f.index.unique())
    train_rows = len(train_data.index.unique())
    val_rows = len(val_data.index.unique())
    test_rows = len(test_data.index.unique())
    print(f"Total number of rows: {total_rows}")
    print(f"Number of training rows: {train_rows} ({100*train_rows/total_rows}%)")
    print(f"Number of validation rows: {val_rows} ({100*val_rows/total_rows}%)")
    print(f"Number of test rows: {test_rows} ({100*test_rows/total_rows}%)")

    return train_data, val_data, test_data


def extract_datasets():
    """
    extract_datasets: Extracts the datasets from the csv file and save processed
        dataframes to disk.

    :return: None
    """
    print(f"Using database prefix: {dataset_prefix}")
    print("Feature variables used:")
    for variable in state_vars:
        print(f" - {variable}")
    print()

    if sampling_window_hours == 1:
        filename = f'../data/{dataset_prefix}ventilatedpatients_hourly.csv'
    else:
        filename = f'../data/{dataset_prefix}ventilatedpatients.csv'

    print(f"Reading csv file from {filename}...")

    # Step 1: Load data.
    print("Step 1: Loading data...")
    df = retrieve_table(filename)
    print("Step 1 complete: Dataframe loaded successfully!\n")

    # Add "unique_id" for the ICU stay based on the database-specific columns.
    if 'stay_id' in df.columns:
        df['unique_id'] = df.stay_id
    elif 'admissionid' in df.columns:
        df['unique_id'] = df.admissionid

    # Step 2: Retrieve indices of relevant mechanical ventilation rows.
    print("Step 2: Retrieving MV indices...")
    unique_ids, mechvent_episodes_indices = retrieve_mv_indices(df)
    print(f"Step 2 completed: {len(mechvent_episodes_indices)} MV indices were retrieved.\n")

    # Step 3: Use forward filling / Sample-and-Hold imputation.
    print("Step 3: Using forward filling / Sample-and-Hold imputation...")
    df = forward_fill(df)
    print("Step 3 completed.\n")

    # Step 4: Filter out only the Mechanical Ventilation rows.
    print("Step 4: Filtering out only the Mechanical Ventilation rows...")
    df_imv = df.iloc[mechvent_episodes_indices, :]
    print(f"Step 4 completed: {len(df_imv.index)} rows remain.\n")

    # Step 5: Remove Outliers
    print("Step 5: Removing Outliers...")
    df_imv_f, selected_patients = remove_outliers(df, df_imv)
    print("Step 5 completed!\n")

    # Step 6: Center and Scale data.
    print("Step 6: Centering and Scaling data...")
    (
        df_imv_f,
        scaled_vars,
        scaled_observational_varnames,
        demographical_scaled_vars,
        scaled_demographical_varnames,
        observational_vars_scaler,
        demographical_vars_scaler
    ) = center_and_scale(df_imv_f)
    print("Step 6 completed!\n")

    # Step 7: Split dataset into Train/Test/Validation splits.
    print("Step 7: Splitting dataset into Train/Test/Validation splits...")
    train_data, val_data, test_data = dataset_splits(df_imv_f)
    print("Step 7 completed!\n")

    # Step 8: Impute remaining missing values with KNN imputation and save tables
    print("Step 8: Imputing remaining missing values with KNN imputation...")
    scaled_varnames = scaled_observational_varnames + scaled_demographical_varnames
    print("Imputing Train set...")
    df_train_final, imputed_vars_train, imputer_train, imputed_varnames_train = knn_imputation(train_data, scaled_varnames, scaled_vars, save_location=f"../data_processors/{dataset_prefix}KNNimputer.bin")
    df_train_final.to_csv(f"../data/{dataset_prefix}train_data.csv")
    print("Imputing Validation set...")
    df_val_final, imputed_vars_val, imputer_val, imputed_varnames_val = knn_imputation(val_data, scaled_varnames, scaled_vars, save_location=f"../data_processors/{dataset_prefix}KNNimputer_val.bin")
    df_val_final.to_csv(f"../data/{dataset_prefix}val_data.csv")
    print("Imputing Test set...")
    df_test_final, imputed_vars_test, imputer_test, imputed_varnames_test = knn_imputation(test_data, scaled_varnames, scaled_vars, save_location=f"../data_processors/{dataset_prefix}KNNimputer_test.bin")
    df_test_final.to_csv(f"../data/{dataset_prefix}test_data.csv")

    print(df_train_final)
    print(df_val_final)
    print(df_test_final)
    print("Step 8 completed!\n")

    # Step 9: Save Scalers and table
    print("Step 9: Saving Scalers and final tables...")
    # df_train_final.to_csv("../data/train_data.csv")
    # df_val_final.to_csv("../data/val_data.csv")
    # df_test_final.to_csv("../data/test_data.csv")
    joblib.dump(scaled_observational_varnames, f'../data_processors/{dataset_prefix}scaled_observational_varnames.bin', compress=True)
    joblib.dump(observational_vars_scaler, f"../data_processors/{dataset_prefix}observational_vars_scaler.bin", compress=True)
    joblib.dump(scaled_demographical_varnames, f'../data_processors/{dataset_prefix}scaled_demographical_varnames.bin', compress=True)
    joblib.dump(demographical_vars_scaler, f"../data_processors/{dataset_prefix}demographical_vars_scaler.bin", compress=True)
    print("Step 9 completed!\n")


if __name__ == '__main__':
    set_seed(42)
    extract_datasets()