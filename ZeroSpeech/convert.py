import hydra
import hydra.utils as utils
import json
from pathlib import Path
import soundfile
import torch
import numpy as np
import librosa
from tqdm import tqdm
import pyloudnorm
from preprocess import preemphasis
from vq_vae.model import Encoder, Decoder, VqVae


@hydra.main(config_path="config", config_name="convert.yaml")
def convert(cfg):
    dataset_path = Path(utils.to_absolute_path("zerospeech2020_datasets")) / cfg.dataset.path
    with open(dataset_path / "speakers.json") as file:
        speakers = sorted(json.load(file))

    synthesis_list_path = Path(utils.to_absolute_path(cfg.synthesis_list))
    with open(synthesis_list_path) as file:
        synthesis_list = json.load(file)

    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VqVae(Encoder(**cfg.model.encoder), Decoder(**cfg.model.decoder))
    model.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    meter = pyloudnorm.Meter(cfg.preprocessing.sr)

    for wav_path, speaker_id, out_filename in tqdm(synthesis_list):
        wav_path = in_dir / wav_path
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),
            sr=cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(wav)
        wav = wav / np.abs(wav).max() * 0.999

        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1
        )
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1

        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device)
        with torch.no_grad():
            z, _ = model.encode(mel)
            output = model.decoder.generate(z, speaker)

        output_loudness = meter.integrated_loudness(output)
        output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
        path = out_dir / out_filename
        soundfile.write(path.with_suffix(".wav"), output.astype(np.float32), samplerate=cfg.preprocessing.sr)


if __name__ == "__main__":
    convert()
