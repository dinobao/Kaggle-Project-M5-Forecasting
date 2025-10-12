# Set directory paths
folder_name = "SUBMISSION MODEL"
dir_ = "E:/5440-prj3/"

raw_data_dir = os.path.join(dir_, "2. data/")
processed_data_dir = os.path.join(dir_, "2. data/processed/")
log_dir = os.path.join(dir_, "4. logs/")
model_dir = os.path.join(dir_, "5. models/")

# Import required libraries
import gc
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Function to optimize memory usage
def memory_optimizer(df, verbose=False):
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]
    initial_memory = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numeric_types:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if np.finfo(np.float16).min < c_min < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif np.finfo(np.float32).min < c_min < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    final_memory = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f"Memory reduced to {final_memory:.2f} MB ({100 * (initial_memory - final_memory) / initial_memory:.1f}% reduction)")
    return df

# Configuration
START_DAY = 710
EXCLUDED_COLUMNS = ["id", "state_id", "store_id", "date", "wm_yr_wk", "d", "sales"]
CATEGORICAL_FEATURES = ["item_id", "dept_id", "cat_id", "state_id", "store_id", "event_name_1", "event_name_2", "event_type_1", "event_type_2"]

GRID2_FEATURES = [
    "sell_price", "price_max", "price_min", "price_std", "price_mean", 
    "price_norm", "price_nunique", "item_nunique", "price_momentum", 
    "price_momentum_m", "price_momentum_y",
]

GRID3_FEATURES = [
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI", "tm_d", "tm_w", "tm_m", 
    "tm_y", "tm_wm", "tm_dw", "tm_w_end",
]

LAG_FEATURES = [
    f"sales_lag_{i}" for i in range(28, 43)
] + [
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
    "rolling_mean_30", "rolling_std_30", "rolling_mean_60", "rolling_std_60",
    "rolling_mean_180", "rolling_std_180",
]

ENCODING_FEATURES = [
    "enc_store_id_dept_id_mean", "enc_store_id_dept_id_std",
    "enc_item_id_state_id_mean", "enc_item_id_state_id_std",
]

# Prepare dataset for a specific store
def load_store_data(store):
    part1 = pd.read_pickle(os.path.join(processed_data_dir, "grid_part_1.pkl"))
    part2 = pd.read_pickle(os.path.join(processed_data_dir, "grid_part_2.pkl"))[GRID2_FEATURES]
    part3 = pd.read_pickle(os.path.join(processed_data_dir, "grid_part_3.pkl"))[GRID3_FEATURES]

    combined = pd.concat([part1, part2, part3], axis=1)
    del part1, part2, part3
    gc.collect()

    combined = combined[combined["store_id"] == store]
    combined = combined[combined["d"] >= START_DAY]

    lags = pd.read_pickle(os.path.join(processed_data_dir, "lags_df_28.pkl"))[LAG_FEATURES]
    lags = lags.loc[combined.index]

    encodings = pd.read_pickle(os.path.join(processed_data_dir, "mean_encoding_df.pkl"))[ENCODING_FEATURES]
    encodings = encodings.loc[combined.index]

    combined = pd.concat([combined, lags, encodings], axis=1)
    combined = memory_optimizer(combined)

    return combined

# Validation split configuration
VALIDATION_SPLITS = {
    "cv1": [1551, 1610],
    "cv2": [1829, 1857],
    "cv3": [1857, 1885],
    "cv4": [1885, 1913],
    "public": [1913, 1941],
    "private": [1941, 1969],
}

# LightGBM parameters
LIGHTGBM_PARAMS = {
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
    "seed": 2024,
}

# Train and save models for each validation split and store
for cv, days in VALIDATION_SPLITS.items():
    print(f"Processing validation split: {cv}...")

    for store in STORES:
        print(f"Training for store: {store}")
        store_data = load_store_data(store)

        features = [col for col in store_data.columns if col not in EXCLUDED_COLUMNS]
        train_mask = (store_data["d"] >= START_DAY) & (store_data["d"] <= days[0])
        valid_mask = (store_data["d"] > days[0]) & (store_data["d"] <= days[1])

        train_dataset = lgb.Dataset(store_data[train_mask][features], label=store_data[train_mask]["sales"])
        valid_dataset = lgb.Dataset(store_data[valid_mask][features], label=store_data[valid_mask]["sales"])

        model = lgb.train(
            LIGHTGBM_PARAMS,
            train_dataset,
            valid_sets=[valid_dataset, train_dataset],
            verbose_eval=100,
        )

        feature_importances = pd.DataFrame({
            "Feature": model.feature_name(),
            "Importance": model.feature_importance(),
        }).sort_values(by="Importance", ascending=False)

        print(feature_importances.head(10))

        model_filename = os.path.join(model_dir, f"non_recursive_{store}_{cv}.bin")
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)

        del store_data, train_dataset, valid_dataset, model
        gc.collect()