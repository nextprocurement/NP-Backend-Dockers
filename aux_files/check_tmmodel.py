import pathlib
from tqdm import tqdm

path_models_destination = pathlib.Path("data/source/cpv_models")

tmmodel_founds = 0
has_content = 0
nr_models = len(list(path_models_destination.iterdir())) 
for directory in tqdm(path_models_destination.iterdir()):
    if not directory.is_dir():
        continue
    # TMmodel exists
    assert (directory / "model_data" / "TMmodel").is_dir(), f"TMmodel not found in {directory}"
    tmmodel_founds += 1
    # TMmodel has content
    assert len(list((directory / "model_data" / "TMmodel").iterdir())) > 0, f"TMmodel empty in {directory}"
    has_content += 1
print(f"TMmodel found in {tmmodel_founds} out of {nr_models} models at model_data/TMmodel")
print(f"TMmodel has content in {has_content} out of {tmmodel_founds} models at model_data/TMmodel")