import os
import glob

def get_most_recent_pt_file(directory_path, extension=".pt"):
    # Search for all ".pt" files in the given directory
    pt_files = glob.glob(os.path.join(directory_path, "*"+extension))

    # Ensure there's at least one file
    if not pt_files:
        return None

    # Find the most recent file
    most_recent_file = max(pt_files, key=os.path.getmtime)

    # Return just the name of the most recent file
    return os.path.basename(most_recent_file)
