import numpy as np
import torch
from random import randint
from pathlib import Path
from kaldi_format_dataset import KaldiFormatDataset


class SpeechDataset(KaldiFormatDataset):
    def __init__(
            self,
            root: Path,
            split_name: str,
            hop_length: int,
            sr: int,
            sample_frames: int,
            include_utts=False,
            subsample=True,
    ):
        super().__init__(root, split_name)

        self.feats_dir = Path(root) / 'feats' / split_name
        self.hop_length = hop_length
        self.sample_frames = sample_frames
        self.subsample = subsample
        self.include_utts = include_utts

        self.data = [
            (utt, Path(self.feats_dir / utt)) for utt, _ in self.utt2spk.items()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utt, path = self.data[index]  # type: str, Path

        audio = np.load(str(path.with_suffix(".wav.npy")))
        mel = np.load(str(path.with_suffix(".mel.npy")))

        if self.subsample:
            pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
            mel = mel[:, pos - 1:pos + self.sample_frames + 1]
            audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = self.speakers.index(self.utt2spk[utt])

        if self.include_utts:
            return utt, torch.LongTensor(audio), torch.FloatTensor(mel), speaker
        else:
            return torch.LongTensor(audio), torch.FloatTensor(mel), speaker
