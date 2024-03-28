import os

def add_suffix_to_folders(folder_path, suffix):
    """
    Add a suffix to all folders in the specified folder.

    Parameters:
    - folder_path (str): Path to the folder containing subfolders.
    - suffix (str): The string suffix to be added to each folder.
    """
    try:
        # Get a list of all items (files and folders) in the specified folder
        items = os.listdir(folder_path)

        # Iterate through each item in the folder
        for item in items:
            item_path = os.path.join(folder_path, item)

            # Check if the item is a folder
            if os.path.isdir(item_path):
                # Add the suffix to the folder name
                new_folder_name = item + suffix

                # Construct the new path for the folder with the added suffix
                new_folder_path = os.path.join(folder_path, new_folder_name)

                # Rename the folder
                os.rename(item_path, new_folder_path)

                print(f'Renamed folder: {item} -> {new_folder_name}')

    except OSError as e:
        print(f"Error: {e}")

# Example Usage:
folder_path = "./data/preprocessed/2000/images/2006/D14/"  # Replace with the path to your folder
suffix = "-0M50-E080"  # Replace with your desired suffix

add_suffix_to_folders(folder_path, suffix)