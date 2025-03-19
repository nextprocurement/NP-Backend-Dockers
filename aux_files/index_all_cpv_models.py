import pathlib
import subprocess
from tqdm import tqdm

path_models_destination = pathlib.Path("data/source/cpv_models")

for directory in tqdm(path_models_destination.iterdir()):
    if not directory.is_dir():
        continue
    
    model_name = f"cpv_models/{directory.name}"
    url = f"http://kumo01:92/models/indexModel/?model_name={model_name}"

    command = [
        "curl",
        "-X", "POST",
        url,
        "-H", "accept: application/json",
        "-d", ""
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)

        print(f"Executed for {model_name}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error for {model_name}: {e}")