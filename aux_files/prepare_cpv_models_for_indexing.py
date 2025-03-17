import json
import pathlib
import shutil

path_models_origin = pathlib.Path("/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/sample_data/final_cpv_models_with_labels_corrected")
path_models_destination = pathlib.Path("data/source/cpv_models")
data_path = "data/source/place_all_embeddings_metadata_only_augmented.parquet"

for version in ["small", "large"]:
    for directory in path_models_origin.iterdir():
        if not directory.is_dir():
            continue

        cpv = directory.name.split("_")[-1] 
        print(f"Processing CPV: {cpv}")

        topics_pos = []
        topics_files = []

        for file in directory.iterdir():
            if file.is_dir():
                try:
                    n_topics = int(file.name.split("_")[0])
                    topics_pos.append(n_topics)
                    topics_files.append(file)
                except ValueError:
                    continue

        if not topics_pos:
            print(f"No valid topics for CPV {cpv}")
            continue

        n_topics_graph = min(topics_pos) if version == "small" else max(topics_pos)
        selected_model_path = topics_files[topics_pos.index(n_topics_graph)]
        
        path_model_save = path_models_destination / f"{cpv}_{version}"
        
        # Copy the content of the selected model to the new directory
        if path_model_save.exists():
            shutil.rmtree(path_model_save)  # Remove existing directory if it exists
        
        shutil.copytree(selected_model_path, path_model_save)
        print(f"Copied {selected_model_path} to {path_model_save}")
        
        # modified the trainconfig to include the local path to the data
        path_config = path_model_save / "trainconfig.json"
        with open(path_config, "r") as f:
            config = json.load(f)
            config["TrDtSet"] = data_path
        # save the modified config
        with open(path_config, "w") as f:
            json.dump(config, f)