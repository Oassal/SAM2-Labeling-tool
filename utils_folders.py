import os
from pathlib import Path

def objects_dict_from_list(objects_names):
    return {name: idx for idx, name in enumerate(objects_names)}

def count_images_in_directory(directory):
    """Count image files in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    # Define the extensions for image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_count = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_count += 1
    return image_count

def list_subfolders(main_folder):
    """
    Returns a list of all subfolders in the given main folder.
    """
    main_path = Path(main_folder)
    return [str(f) for f in main_path.iterdir() if f.is_dir()]