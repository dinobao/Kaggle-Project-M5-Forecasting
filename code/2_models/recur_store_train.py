# Set up directories
folder_name = "SUBMISSION MODEL"
dir_ = "E:/5440-prj3/"

raw_data_dir = os.path.join(dir_, "2. data/")
processed_data_dir = os.path.join(dir_, "2. data/processed/")
log_dir = os.path.join(dir_, "4. logs/")
model_dir = os.path.join(dir_, "5. models/")
submission_dir = os.path.join(dir_, "6. submissions/")

# Import necessary libraries
import numpy as np
import pandas as pd
import os, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Function to set random seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

# Function for parallel processing
def parallelize_data_processing(func, data_split):
    cores = min(psutil.cpu_count(), len(data_split))
    with Pool(cores) as pool:
        data = pd.concat(pool.map(func, data_split), axis=1)
    return data

# Load and filter data for a specific store
def load_store_data(store_id):
    base_df = pd.concat(
        [
            pd.read_pickle(BASE),
            pd.read_pickle(PRICE).iloc[:, 2:],
            pd.read_pickle(CALENDAR).iloc[:, 2:],
        ],
        axis=1,
    )

    base_df = base_df[base_df["d"] >= START_TRAIN]
    base_df = base_df[base_df["store_id"] == store_id]

    mean_enc_data = pd.read_pickle(MEAN_ENC)[mean_features]
    mean_enc_data = mean_enc_data.loc[base_df.index]

    lag_features = pd.read_pickle(LAGS).iloc[:, 3:]
    lag_features = lag_features.loc[base_df.index]

    base_df = pd.concat([base_df, mean_enc_data, lag_features], axis=1)
    del mean_enc_data, lag_features

    valid_features = [
        col for col in base_df.columns if col not in remove_features
    ]
    return base_df.reset_index(drop=True), valid_features

# Prepare test set by recombining data for all stores
def prepare_test_data():
    combined_test_data = pd.DataFrame()
    for store in STORES_IDS:
        test_data = pd.read_pickle(os.path.join(processed_data_dir, f"test_{store}.pkl"))
        test_data["store_id"] = store
        combined_test_data = pd.concat([combined_test_data, test_data])
    return combined_test_data.reset_index(drop=True)

# Helper to create lag features
def generate_lag_feature(lag):
    temp_lag_df = base_test.copy()[["id", "d", TARGET]]
    lag_column = f"sales_lag_{lag}"
    temp_lag_df[lag_column] = (
        temp_lag_df.groupby("id")[TARGET].shift(lag).astype(np.float16)
    )
    return temp_lag_df[[lag_column]]

# Helper to create rolling features
def generate_rolling_feature(shift_window_pair):
    shift_days, window_size = shift_window_pair
    temp_roll_df = base_test.copy()[["id", "d", TARGET]]
    rolling_column = f"rolling_mean_{shift_days}_{window_size}"
    temp_roll_df[rolling_column] = (
        temp_roll_df.groupby("id")[TARGET]
        .shift(shift_days)
        .rolling(window_size)
        .mean()
    )
    return temp_roll_df[[rolling_column]]

# Model parameters
lgb_params = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "subsample": 0.5,
    "subsample_freq": 1,
    "learning_rate": 0.015,
    "num_leaves": 2047,
    "min_data_in_leaf": 4095,
    "feature_fraction": 0.5,
    "max_bin": 100,
    "n_estimators": 3000,
    "boost_from_average": False,
    "verbose": -1,
}

# General settings
VER = 1
SEED = 42
set_seed(SEED)

TARGET = "sales"
START_TRAIN = 0
P_HORIZON = 28
USE_AUX = False

remove_features = [
    "id",
    "state_id",
    "store_id",
    "date",
    "wm_yr_wk",
    "d",
    TARGET,
]
mean_features = [
    "enc_cat_id_mean",
    "enc_cat_id_std",
    "enc_dept_id_mean",
    "enc_dept_id_std",
    "enc_item_id_mean",
    "enc_item_id_std",
]

# File paths
BASE = os.path.join(processed_data_dir, "grid_part_1.pkl")
PRICE = os.path.join(processed_data_dir, "grid_part_2.pkl")
CALENDAR = os.path.join(processed_data_dir, "grid_part_3.pkl")
LAGS = os.path.join(processed_data_dir, "lags_df_28.pkl")
MEAN_ENC = os.path.join(processed_data_dir, "mean_encoding_df.pkl")

# Create lag and rolling splits
LAGS_SPLIT = list(range(28, 28 + 15))
ROLS_SPLIT = [[i, j] for i in [1, 7, 14] for j in [7, 14, 30, 60]]

_, MODEL_FEATURES = load_store_data(STORES_IDS[-1])
del _

# Main prediction loop
all_predictions = pd.DataFrame()
base_test = prepare_test_data()

for PREDICT_DAY in range(1, 29):
    print(f"Predicting Day {PREDICT_DAY}")
    grid_df = base_test.copy()

    # Generate rolling features
    rolling_features = []
    for roll in ROLS_SPLIT:
        rolling_features.append(generate_rolling_feature(roll))
    rolling_features_df = pd.concat(rolling_features, axis=1)
    grid_df = pd.concat([grid_df, rolling_features_df], axis=1)

    # Predict for each store
    for store in STORES_IDS:
        model_path = os.path.join(model_dir, f"lgb_model_{store}_v{VER}.bin")
        model = pickle.load(open(model_path, "rb"))

        prediction_mask = (base_test["d"] == END_TRAIN + PREDICT_DAY) & (
            base_test["store_id"] == store
        )
        base_test.loc[prediction_mask, TARGET] = model.predict(
            grid_df.loc[prediction_mask, MODEL_FEATURES]
        )

    # Store predictions
    day_prediction = base_test.loc[
        base_test["d"] == END_TRAIN + PREDICT_DAY, ["id", TARGET]
    ]
    day_prediction.columns = ["id", f"F{PREDICT_DAY}"]
    all_predictions = pd.merge(
        all_predictions, day_prediction, on="id", how="outer"
    )

# Export predictions
submission = pd.read_csv(os.path.join(raw_data_dir, "sample_submission.csv"))[["id"]]
submission = submission.merge(all_predictions, on="id", how="left").fillna(0)
submission.to_csv(
    os.path.join(submission_dir, "before_ensemble", "submission_kaggle_recursive_store.csv"),
    index=False,
)