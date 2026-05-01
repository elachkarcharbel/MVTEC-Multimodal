import os
import shutil
import pandas as pd

# Paths to the original CSVs
old_train_csv = '../binary_labels/train.csv'
old_test_csv = '../binary_labels/test.csv'

# Base path for the image folders
base_folder = '../MVTEC-AD-Original-Normalized700x700'

# Paths to the new CSVs
new_train_csv = '../binary_labels/new_train.csv'
new_test_csv = '../binary_labels/new_test.csv'

# Paths to the new folders
new_train_folder = os.path.join(base_folder, 'new_train')
new_test_folder = os.path.join(base_folder, 'new_test')

# Create new folders if they don't exist
os.makedirs(new_train_folder, exist_ok=True)
os.makedirs(new_test_folder, exist_ok=True)

# Read the old CSVs
old_train_df = pd.read_csv(old_train_csv)
old_test_df = pd.read_csv(old_test_csv)

# Combine the dataframes
combined_df = pd.concat([old_train_df, old_test_df], ignore_index=True)

# Shuffle the combined dataframe
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

# Split into new training and testing sets (80% train, 20% test)
split_index = int(0.8 * len(shuffled_df))
new_train_df = shuffled_df[:split_index]
new_test_df = shuffled_df[split_index:]

# Save the new CSVs
new_train_df.to_csv(new_train_csv, index=False)
new_test_df.to_csv(new_test_csv, index=False)

# Function to copy images based on the new CSVs
def copy_images(df, base_folder, new_folder):
    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        # Check if img_name already contains 'train/' or 'test/'
        if 'train/' in img_name or 'test/' in img_name:
            src_path = os.path.join(base_folder, img_name)
            img_name = os.path.basename(img_name)  # only keep the image name
        else:
            src_path_train = os.path.join(base_folder, 'train', img_name)
            src_path_test = os.path.join(base_folder, 'test', img_name)
            if os.path.exists(src_path_train):
                src_path = src_path_train
            elif os.path.exists(src_path_test):
                src_path = src_path_test
            else:
                print(f"Warning: {img_name} does not exist in either training or testing folders.")
                continue
        
        dest_path = os.path.join(new_folder, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)

# Copy images to new folders using updated function
copy_images(new_train_df, base_folder, new_train_folder)
copy_images(new_test_df, base_folder, new_test_folder)

print(f"New training set created with {len(new_train_df)} images.")
print(f"New testing set created with {len(new_test_df)} images.")
