import hydra
import hydra.utils as utils
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from model import Encoder, Decoder, VqVae


def get_dataloader(cfg):
    from torch.utils.data import DataLoader
    from dataset import SpeechDataset

    root_path = Path(utils.to_absolute_path(cfg.in_dir))
    dataset = SpeechDataset(
        root=root_path,
        split_name=cfg.data_split,
        hop_length=cfg.preprocessing.hop_length,
        sample_frames=cfg.training.sample_frames,
        include_utts=True,
        subsample=False,
    )

    return DataLoader(
        dataset,
        batch_size=1,
    )


@hydra.main(config_path="config", config_name="encode.yaml")
def encode(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VqVae(Encoder(**cfg.model.encoder), Decoder(**cfg.model.decoder))
    model.to(device)

    print(f"Load checkpoint from: {cfg.checkpoint}")
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataloader = get_dataloader(cfg)
    for _, (utts, audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
        wavs, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

        with torch.no_grad():
            z, idx = model.encode(mels)

        n = z.shape[0]
        for i in range(n):
            output = z[i].detach().cpu().numpy()

            out_path = out_dir / f'{utts[i]}.npy'
            np.save(str(out_path), output)


if __name__ == "__main__":
    encode()
