import os
import glob
import pathlib
import json
import shutil


TARGET_SEARCH_DIR = "/home/dengarden/Documents/AI_project/vanila_model/AlphaFold_prediction/deprecated/on_10"
OUTPUT_DIR = "/mnt/P41/Repositories/ProteinMPNN/AI_project/plddt_analysis_target/"
# JSON listing
# Search recursively
for filename in sorted(
    glob.iglob(f"{TARGET_SEARCH_DIR}/**/ranking_debug.json", recursive=True)
):
    p = os.path.abspath(filename)

    # DEBUG
    # print(p)
    # print(pathlib.Path(p).parent.name)

    sample_name = pathlib.Path(p).parent.name
    # Read JSON
    with open(p) as j:
        data = json.load(j)

        # DEV
        # print(data["order"][0])

        target_pkl = f"{pathlib.Path(p).parent}/result_{data['order'][0]}.pkl"

        shutil.copy(target_pkl, f"{OUTPUT_DIR}/{sample_name}.pkl")

# Pick best prediction

# then copy it
