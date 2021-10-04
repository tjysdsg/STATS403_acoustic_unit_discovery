import hydra
import hydra.utils as utils
from matplotlib import pyplot as plt
import numpy as np
import os


@hydra.main(config_path="config", config_name="visualize.yaml")
def main(cfg):
    in_dir = utils.to_absolute_path(cfg.in_dir)
    out_dir = utils.to_absolute_path(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(in_dir):
        path = os.path.join(in_dir, f)
        if os.path.isfile(path):
            utt = f.split('.')[0]

            encoding: np.ndarray = np.load(path)

            plt.figure(figsize=(19, 10))
            plt.imshow(encoding.T)
            plt.xlabel('Sub-sampled frames')
            plt.ylabel('Encodings')
            plt.savefig(os.path.join(out_dir, f'{utt}.png'))
            plt.close('all')


if __name__ == '__main__':
    main()
