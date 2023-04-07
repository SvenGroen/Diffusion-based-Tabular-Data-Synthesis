import os
from pathlib import Path

def _strip_path_to_src_parent(path, target_subfolder):
    path_parts = path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part == target_subfolder and i > 0:
            stripped_path = os.path.join(*path_parts[:i])
            return stripped_path
    # if this is reached, the target subfolder was not found in the path
    # --> make sure it is exists underneath the path
    if not os.path.exists(os.path.join(path, target_subfolder)):
        raise ValueError(f"The folder '{target_subfolder}' was not found in the path '{path}', nor is it a subfolder of the path.\
                         Make sure the Current Working Directory is set to the root of the project.")
    else:
        return path
    

RUNS_IN_CLOUD = os.getenv("AZUREML_RUN_ID") is not None
CWD = os.getcwd()
# This ensures that the CWD is set to the root of the project, as long as the scripts are run from any subfolder of the project.
# This was necessary when training with azure,
#       CWD_in_Azure=PROJEKT_FOLDER/src/
#       CWD_in_local=PROJEKT_FOLDER/
# This could be easy fixable by running the script inside azure from PROJEKT_FOLDER/ as CWD,
# however, this would require addind  PROJEKT_FOLDER/src to the PYTHONPATH in the azure script.

if RUNS_IN_CLOUD:
    ROOT_DIR = CWD
else:
    ROOT_DIR = Path("src")