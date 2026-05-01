import pandas as pd
import os

def copy_and_prefix_images(train_template_csv, test_template_csv, initial_train_csv, initial_test_csv, output_train_csv, output_test_csv):
    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_csv), exist_ok=True)

    # Read the template CSVs
    train_template_df = pd.read_csv(train_template_csv)
    test_template_df = pd.read_csv(test_template_csv)

    # Read the initial CSV files
    initial_train_df = pd.read_csv(initial_train_csv)
    initial_test_df = pd.read_csv(initial_test_csv)

    # Combine initial train and test dataframes for easier processing
    initial_df = pd.concat([initial_train_df, initial_test_df])
    initial_df['image_name'] = initial_df['image_path'].apply(lambda x: x.split('/')[-1])
    print(initial_df.head())

    # Create a copy of image_path without prefixes for matching
    train_template_df['image_name'] = train_template_df['image_path'].apply(lambda x: x.split('/')[-1])
    test_template_df['image_name'] = test_template_df['image_path'].apply(lambda x: x.split('/')[-1])

    # Merge the initial dataframe with the template dataframes based on image_name
    train_merged_df = train_template_df[['image_path', 'image_name', 'label']].merge(initial_df[['image_name', 'caption']], on='image_name', how='left')
    test_merged_df = test_template_df[['image_path', 'image_name', 'label']].merge(initial_df[['image_name', 'caption']], on='image_name', how='left')

    # Ensure only required columns are kept
    train_merged_df = train_merged_df[['image_path', 'caption', 'label']]
    test_merged_df = test_merged_df[['image_path', 'caption', 'label']]

    # Save the new CSV files
    train_merged_df.to_csv(output_train_csv, index=False)
    test_merged_df.to_csv(output_test_csv, index=False)
    
    print("Final Train DataFrame saved to:", output_train_csv)
    print(train_merged_df.head())
    print("Final Test DataFrame saved to:", output_test_csv)
    print(test_merged_df.head())

def check_csv_matching(template_csv, generated_csv):
    # Read the template and generated CSV files
    template_df = pd.read_csv(template_csv)
    generated_df = pd.read_csv(generated_csv)

    # Compare the full image_path in both dataframes
    match = template_df['image_path'].equals(generated_df['image_path'])

    if match:
        print(f"Success: The {os.path.basename(template_csv)} matches the {os.path.basename(generated_csv)} in terms of full image_path.")
    else:
        print(f"Error: The {os.path.basename(template_csv)} does not match the {os.path.basename(generated_csv)} in terms of full image_path.")

# Paths to your CSV files
model_name = 'CLIP(ViT-L-14+Conceptual)+GPT2'
train_template_csv = '../binary_labels/new_train.csv'
test_template_csv = '../binary_labels/new_test.csv'
initial_train_csv = '../MVTEC-AD-Captions/CLIP/'+model_name+'/train.csv'
initial_test_csv = '../MVTEC-AD-Captions/CLIP/'+model_name+'/test.csv'
output_train_csv = '../binary_labels/CLIP/'+model_name+'/new_train.csv'
output_test_csv = '../binary_labels/CLIP/'+model_name+'/new_test.csv'

# Copy and add prefixes to the images
copy_and_prefix_images(train_template_csv, test_template_csv, initial_train_csv, initial_test_csv, output_train_csv, output_test_csv)

# Check if the template and generated CSV files match
check_csv_matching(train_template_csv, output_train_csv)
check_csv_matching(test_template_csv, output_test_csv)
