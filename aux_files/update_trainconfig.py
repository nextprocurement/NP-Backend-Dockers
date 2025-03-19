import json
import pathlib

path_models_destination = pathlib.Path("data/source/cpv_models")
data_path = "data/source/place_all_embeddings_metadata.parquet"

for directory in path_models_destination.iterdir():
    if not directory.is_dir():
        continue
    
    path_config = directory / "trainconfig.json"
    with open(path_config, "r") as f:
        config = json.load(f)
        config["TrDtSet"] = data_path
    # save the modified config
    with open(path_config, "w") as f:
        json.dump(config, f)