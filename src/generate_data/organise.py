import os
import shutil

def organize_images_by_number(source_folder, dest_folder):
    # Iterate over the files in the source folder
    for filename in os.listdir(source_folder):
        # Extract CW ID and image name
        extract_cw_id = filename.split('_CW_')[-1].split('_')[0]
        
        # Move the file to the corresponding folder
        target_dir = os.path.join(dest_folder,extract_cw_id)
        #print(target_dir)
        if not os.path.isdir(target_dir):
            #print(f"Creating folder: {target_dir}")
            os.makedirs(target_dir)

        shutil.copy(
            os.path.join(source_folder, filename),
            os.path.join(target_dir, filename)
        )