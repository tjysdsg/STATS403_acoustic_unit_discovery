import hydra
import hydra.utils as utils
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from vq_vae.model import Encoder, Decoder, VqVae


@hydra.main(config_path="config", config_name="encode.yaml")
def encode_dataset(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    root_path = Path(utils.to_absolute_path("zerospeech2020_datasets")) / cfg.dataset.path
    with open(root_path / "test.json") as file:
        metadata = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VqVae(Encoder(**cfg.model.encoder), Decoder(**cfg.model.decoder))
    model.to(device)

    print(f"Load checkpoint from: {cfg.checkpoint}:")
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for _, _, _, path in tqdm(metadata):
        path = root_path.parent / path
        mel = torch.from_numpy(np.load(path.with_suffix(".mel.npy")).astype('float32')).unsqueeze(0).to(device)
        with torch.no_grad():
            z, indices = model.encode(mel)

        z = z.squeeze().cpu().numpy()

        out_path = out_dir / path.stem
        with open(out_path.with_suffix(".txt"), "w") as file:
            np.savetxt(file, z, fmt="%.16f")


if __name__ == "__main__":
    encode_dataset()
