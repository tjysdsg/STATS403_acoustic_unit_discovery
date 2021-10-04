import hydra
from hydra import utils
from pathlib import Path
import librosa
import scipy
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def process_wav(
        wav_path: Path, out_path: Path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
        win_length=400, fmin=50, top_db=80, bits=8
):
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2 ** bits)

    np.save(str(out_path.with_suffix(".wav.npy")), wav)
    np.save(str(out_path.with_suffix(".mel.npy")), logmel)
    return out_path, logmel.shape[-1]


@hydra.main(config_path='config', config_name='preprocessing')
def preprocess_dataset(cfg):
    in_dir = Path(utils.to_absolute_path(cfg.data_dir))
    out_dir = Path(utils.to_absolute_path(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    for split in ["train", "l2arctic_test"]:
        print("Extracting features for {} set".format(split))
        futures = []
        split_path = in_dir / split
        out_split = out_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        with open(split_path / "wav.scp") as file:
            for line in file:
                utt, wav_path = line.split()
                out_path = out_split / utt
                futures.append(executor.submit(
                    partial(process_wav, Path(wav_path), out_path, **cfg.preprocessing)
                ))

        results = [future.result() for future in tqdm(futures)]

        lengths = [x[-1] for x in results]
        frames = sum(lengths)
        frame_shift_ms = cfg.preprocessing.hop_length / cfg.preprocessing.sr
        hours = frames * frame_shift_ms / 3600
        print(f"Wrote {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")


if __name__ == "__main__":
    preprocess_dataset()
