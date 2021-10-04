from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Any


class KaldiFormatDataset(Dataset):
    def __init__(self, root: Path, data_split: str):
        super().__init__()
        self.root = Path(root) / data_split

        self.utt2spk = self._read_utt2sth(self.root / 'utt2spk')
        self.spk2utt = self._read_utt2seq(self.root / 'spk2utt')
        self.speakers = sorted(list(set(self.utt2spk.values())))

    def _read_utt2sth(self, path, formatter: Callable[[str], Any] = str):
        ret = {}
        with open(path) as f:
            for line in f:
                utt, sth = line.strip('\n').split(maxsplit=1)
                ret[utt] = formatter(sth)
        return ret

    def _read_utt2seq(self, path, formatter: Callable[[str], Any] = str):
        ret = {}
        with open(path) as f:
            for line in f:
                tokens = line.strip('\n').split()
                utt = tokens[0]
                seq = [formatter(t) for t in tokens[1:]]
                ret[utt] = seq
        return ret

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
