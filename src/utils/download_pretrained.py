import sys
from pathlib import Path

import gdown


def download_best_model(path, link, config_link):
    dir = Path(path)
    dir.mkdir(exist_ok=True)
    model_path = dir / "checkpoint.pth"
    config_path = dir / "config.yaml"
    if not model_path.exists():
        gdown.download(link, str(model_path))
    if not config_path.exists():
        if config_link != "":
            gdown.download(config_link, str(config_path))


if __name__ == "__main__":
    download_best_model(sys.argv[1], sys.argv[2], sys.argv[3])
