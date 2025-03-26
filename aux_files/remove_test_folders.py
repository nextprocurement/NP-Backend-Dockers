import pathlib
import subprocess
from tqdm import tqdm

path_models_destination = pathlib.Path("data/source/cpv_models")

for directory in tqdm(path_models_destination.iterdir()):
    if not directory.is_dir():
        continue
    if (directory / "test_data").is_dir():
        subprocess.run(["rm", "-r", str(directory / "test_data")])
        print(f"Removed test_data from {directory}")
