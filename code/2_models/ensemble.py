# Define directory paths
# test_0410
folder_name = "SUBMISSION MODEL"
dir_ = "E:/5440-prj3/"

raw_data_dir = os.path.join(dir_, "2. data/")
processed_data_dir = os.path.join(dir_, "2. data/processed/")
log_dir = os.path.join(dir_, "4. logs/")
model_dir = os.path.join(dir_, "5. models/")
submission_dir = os.path.join(dir_, "6. submissions/")

# Import necessary libraries
import pandas as pd

# Load the base submission file
submission_data = pd.read_csv(os.path.join(raw_data_dir, "sample_submission.csv"))
submission_ids = pd.DataFrame({"id": submission_data.iloc[30490:]["id"]})

# Load individual submission files
recursive_submission = pd.read_csv(
    os.path.join(submission_dir, "before_ensemble/submission_kaggle_recursive_store.csv")
)
non_recursive_submission = pd.read_csv(
    os.path.join(submission_dir, "before_ensemble/submission_kaggle_nonrecursive_store.csv")
)

# Merge submission files with the filtered IDs
recursive_submission = submission_ids.merge(
    recursive_submission, on="id", how="left"
).set_index("id")
non_recursive_submission = submission_ids.merge(
    non_recursive_submission, on="id", how="left"
).set_index("id")

# Combine submissions with equal weighting
final_submission = (recursive_submission + non_recursive_submission) / 2

# Save the final submission file
final_submission_path = os.path.join(submission_dir, "submission_final.csv")
final_submission.to_csv(final_submission_path)

# Display the final submission
final_submission