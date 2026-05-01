import os
import shutil
import pandas as pd

# Paths to the original CSVs
old_train_csv = '../binary_labels/new_train.csv'
old_test_csv = '../binary_labels/new_test.csv'

# Base path for the image folders
base_folder = '../MVTEC-AD-TIR'

# Paths to the new CSVs
new_train_csv = '../binary_labels/new_train_TIR.csv'
new_test_csv = '../binary_labels/new_test_TIR.csv'

# Paths to the new folders
new_train_folder = os.path.join(base_folder, 'new_train_TIR')
new_test_folder = os.path.join(base_folder, 'new_test_TIR')

# Create new folders if they don't exist
os.makedirs(new_train_folder, exist_ok=True)
os.makedirs(new_test_folder, exist_ok=True)

# Function to modify image names
def modify_image_names(df):
    df['image_path'] = df['image_path'].apply(lambda x: x.replace('.png', '.png'))
    return df

# Read and modify the old CSVs
old_train_df = pd.read_csv(old_train_csv)
old_test_df = pd.read_csv(old_test_csv)
new_train_df = modify_image_names(old_train_df)
new_test_df = modify_image_names(old_test_df)

# Save the new CSVs
new_train_df.to_csv(new_train_csv, index=False)
new_test_df.to_csv(new_test_csv, index=False)

# Function to copy images based on the new CSVs
def copy_images(df, base_folder, new_folder):
    for idx, row in df.iterrows():
        img_name = row['image_path']
        img_name_DM = img_name.replace('.png', '.png') if not img_name.endswith('.png') else img_name

        # Search for the image in both 'train' and 'test' folders
        src_path_train = os.path.join(base_folder, 'train', os.path.basename(img_name_DM))
        src_path_test = os.path.join(base_folder, 'test', os.path.basename(img_name_DM))

        if os.path.exists(src_path_train):
            src_path = src_path_train
        elif os.path.exists(src_path_test):
            src_path = src_path_test
        else:
            print(f"Warning: {img_name_DM} does not exist in either training or testing folders.")
            continue

        # Ensure the destination directory exists and then copy the file
        dest_path = os.path.join(new_folder, os.path.basename(img_name_DM))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        except Exception as e:
            print(f"Error copying {src_path} to {dest_path}: {e}")

# Copy images to new folders using the updated function
copy_images(new_train_df, base_folder, new_train_folder)
copy_images(new_test_df, base_folder, new_test_folder)

