import os
import shutil
from pathlib import Path

def copy_yaml_to_new_folder(source_dir: str, target_dir: str):
    """
    Create a new empty folder and copy all .yaml files from the source folder to the new folder.

    :param source_dir: Path to the source folder
    :param target_dir: Path to the target folder
    """
    source_dir = os.path.join(source_dir, "kustomize", "components", "network-policies")
    target_dir = os.path.join(target_dir, "policies")
    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    # If the target directory exists, remove it and create a new empty folder
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Find and copy all .yaml files to the target directory
    yaml_files = Path(source_dir).glob("*.yaml")
    for yaml_file in yaml_files:
        shutil.copy(yaml_file, target_dir)



