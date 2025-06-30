import os
import shutil

def organize_images_by_number(source_folder, dest_folder):
    # Iterate over the files in the source folder
    for filename in os.listdir(source_folder):
        # Extract CW ID and image name
        extract_cw_id = filename.split('_CW_')[-1].split('_')[0]
        image_name = '_'.join(filename.split('_CW_')[0].split('_')[1:-1])

        # Debug: Print the filename, extracted CW ID, and image name
        if filename in ['0-0_kebs108-25_page_1.jpg_245_CW_4_0.png', '0-0_10_English_Nontail_book 2024-25-31_page_1.jpg_117_CW_422_0.png']:
            print(f"Processing: {filename}")
            print(f"Extracted CW ID: {extract_cw_id}")
            print(f"Image name: {image_name}")

        # Move the file to the corresponding folder
        target_dir = os.path.join(dest_folder, image_name, extract_cw_id)
        #print(target_dir)
        if not os.path.isdir(target_dir):
            #print(f"Creating folder: {target_dir}")
            os.makedirs(target_dir)

        shutil.copy(
            os.path.join(source_folder, filename),
            os.path.join(target_dir, filename)
        )

# Example usage

import os
import shutil

def copy_depth2_folders(source_folder, dest_folder):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    for root, dirs, files in os.walk(source_folder):
        depth = root[len(source_folder):].count(os.sep)
        
        if depth == 2:
            folder_name = os.path.basename(root)
            dest_path = os.path.join(dest_folder, folder_name)
            #print(f"Copying folder: {root} to {dest_path}")
            shutil.copytree(root, dest_path, dirs_exist_ok=True)



source_folder = '/data/textar_outputs/word_only'
dest_folder = '/data/textar_outputs/test_augmented_2'

organize_images_by_number(source_folder, dest_folder)

source_folder = '/data/textar_outputs/test_augmented_2'
dest_folder = '/data/textar_outputs/word_cw'


copy_depth2_folders(source_folder, dest_folder)
