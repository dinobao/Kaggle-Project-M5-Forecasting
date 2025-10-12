# Directory setup
folder_name = "SUBMISSION MODEL"
dir_ = "E:/5440-prj3/"

raw_data_dir = os.path.join(dir_, "2. data/")
processed_data_dir = os.path.join(dir_, "2. data/processed/")
log_dir = os.path.join(dir_, "4. logs/")
model_dir = os.path.join(dir_, "5. models/")
submission_dir = os.path.join(dir_, "6. submissions/")

# Imports
import gc
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Memory optimization
def optimize_memory(df, verbose=False):
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numeric_types:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f"Memory reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df

# Configuration
FIRST_DAY = 710
REMOVE_COLS = ["id", "state_id", "store_id", "date", "wm_yr_wk", "d", "sales"]

GRID2_COLUMNS = [
    "sell_price",
    "price_max",
    "price_min",
    "price_std",
    "price_mean",
    "price_norm",
    "price_nunique",
    "item_nunique",
    "price_momentum",
    "price_momentum_m",
    "price_momentum_y",
]

GRID3_COLUMNS = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "tm_d",
    "tm_w",
    "tm_m",
    "tm_y",
    "tm_wm",
    "tm_dw",
    "tm_w_end",
]

LAG_COLUMNS = [
    f"sales_lag_{i}" for i in range(28, 43)
] + [
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_30",
    "rolling_std_30",
    "rolling_mean_60",
    "rolling_std_60",
    "rolling_mean_180",
    "rolling_std_180",
]

MEAN_ENC_COLUMNS = [
    "enc_store_id_dept_id_mean",
    "enc_store_id_dept_id_std",
    "enc_item_id_state_id_mean",
    "enc_item_id_state_id_std",
]

# Function to prepare the dataset for each store
def load_and_prepare_data(store):
    grid_part1 = pd.read_pickle(os.path.join(processed_data_dir, "grid_part_1.pkl"))
    grid_part2 = pd.read_pickle(os.path.join(processed_data_dir, "grid_part_2.pkl"))[GRID2_COLUMNS]
    grid_part3 = pd.read_pickle(os.path.join(processed_data_dir, "grid_part_3.pkl"))[GRID3_COLUMNS]

    combined_data = pd.concat([grid_part1, grid_part2, grid_part3], axis=1)
    combined_data = combined_data[combined_data["store_id"] == store]
    combined_data = combined_data[combined_data["d"] >= FIRST_DAY]

    lags = pd.read_pickle(os.path.join(processed_data_dir, "lags_df_28.pkl"))[LAG_COLUMNS]
    lags = lags.loc[combined_data.index]

    mean_encodings = pd.read_pickle(os.path.join(processed_data_dir, "mean_encoding_df.pkl"))[MEAN_ENC_COLUMNS]
    mean_encodings = mean_encodings.loc[combined_data.index]

    combined_data = pd.concat([combined_data, lags, mean_encodings], axis=1)
    combined_data = optimize_memory(combined_data)
    
    return combined_data

# Cross-validation setup
VALIDATION_SPLITS = {
    "cv1": [1551, 1610],
    "cv2": [1829, 1857],
    "cv3": [1857, 1885],
    "cv4": [1885, 1913],
    "public": [1913, 1941],
    "private": [1941, 1969],
}

# Model configuration
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "subsample": 0.5,
    "subsample_freq": 1,
    "learning_rate": 0.015,
    "num_leaves": 255,
    "min_data_in_leaf": 255,
    "feature_fraction": 0.5,
    "max_bin": 100,
    "n_estimators": 3000,
    "boost_from_average": False,
    "verbose": -1,
}

# Train and predict
for cv_name, days in VALIDATION_SPLITS.items():
    print(f"Processing {cv_name} split...")

    for store in STORES:
        print(f"Processing store {store}...")
        store_data = load_and_prepare_data(store)

        train_mask = (store_data["d"] <= days[0]) & (store_data["d"] >= FIRST_DAY)
        valid_mask = (store_data["d"] > days[0]) & (store_data["d"] <= days[1])

        features = [col for col in store_data.columns if col not in REMOVE_COLS]

        model_path = os.path.join(model_dir, f"non_recur_model_{store}.bin")
        model = pickle.load(open(model_path, "rb"))

        predictions = pd.DataFrame(
            {"id": store_data[valid_mask]["id"], "predicted": model.predict(store_data[valid_mask][features])}
        )

        predictions.to_csv(
            os.path.join(log_dir, f"predictions_{store}_{cv_name}.csv"), index=False
        )

# Submission creation
os.chdir(log_dir)
csv_files = [f for f in os.listdir() if "predictions" in f]
submission = pd.read_csv(os.path.join(raw_data_dir, "sample_submission.csv")).set_index("id")

for file in csv_files:
    temp = pd.read_csv(file)
    temp.set_index("id", inplace=True)
    submission = submission.add(temp, fill_value=0)

submission.reset_index().to_csv(os.path.join(submission_dir, "final_submission.csv"), index=False)