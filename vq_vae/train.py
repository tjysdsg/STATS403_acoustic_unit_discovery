import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Encoder, Decoder


def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


def get_dataloader(cfg):
    from dataset import SpeechDataset

    root_path = Path(utils.to_absolute_path("datasets"))

    dataset = SpeechDataset(
        root=root_path,
        split_name=cfg.data_split,
        hop_length=cfg.preprocessing.hop_length,
        sample_frames=cfg.training.sample_frames,
        l1_only=cfg.l1_only,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True
    )


@hydra.main(config_path="config", config_name="train.yaml")
def train_model(cfg):
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.tensorboard:
        tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=cfg.training.optimizer.lr)
    [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    dataloader = get_dataloader(cfg)

    n_epochs = cfg.training.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            z, vq_loss, perplexity = encoder(mels)
            output = decoder(audio[:, :-1], z, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, amp,
                    scheduler, global_step, checkpoint_dir)

        if writer:
            writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
            writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
            writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perplexity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))


if __name__ == "__main__":
    train_model()
