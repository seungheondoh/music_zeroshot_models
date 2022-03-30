import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import wandb
import torchaudio

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from model.modules import CNN1D, MusicTaggingTransformer
from model.emb_model import EmbModel
from model.lightning_model import ZSLRunner
from loader.dataloader import DataPipeline


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 
    
def get_checkpoint_callback(save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_early_stop_callback() -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min"
    )
    return early_stop_callback

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.task_type}/{args.emb_type}/{args.backbone}/{args.supervisions}/"
    save_hparams(args, save_path)

    wandb.init(config=args)
    wandb.run.name = f"exp/{args.task_type}/{args.emb_type}/{args.backbone}/{args.supervisions}/"
    args = wandb.config

    pipeline = DataPipeline(
                    data_dir = args.data_dir,
                    msd_dir = args.msd_dir,
                    task_type = args.task_type,
                    emb_type = args.emb_type,
                    supervisions = args.supervisions,
                    duration = args.duration,
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers
                    )
    if args.backbone == "CNN1D":
        backbone = CNN1D()
        model = EmbModel(
                audio_model = backbone,
                projection_ndim = 100
        )
    elif args.backbone == "Transformer":
        backbone = MusicTaggingTransformer(conv_ndim=128, attention_ndim=64)
        model = EmbModel(
                audio_model = backbone,
                projection_ndim = 64
        )
    runner = ZSLRunner(
            model = model,
            margin = args.margin, 
            lr = args.lr, 
            supervisions = args.supervisions,
            opt_type = args.opt_type
    )

    logger = get_wandb_logger(model)
    checkpoint_callback = get_checkpoint_callback(save_path)
    early_stop_callback = get_early_stop_callback()
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes=args.num_nodes,
                    gpus= args.gpus,
                    strategy=DDPPlugin(find_unused_parameters=False),
                    logger=logger,
                    callbacks=[
                        early_stop_callback,
                        checkpoint_callback
                    ],
                )

    trainer.fit(runner, datamodule=pipeline)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", default="CNN1D", type=str)
    parser.add_argument("--tid", default="debug", type=str)
    # pipeline
    parser.add_argument("--data_dir", default="../dataset", type=str)
    parser.add_argument("--msd_dir", default="../../media/chopin21/msd_resample", type=str)
    parser.add_argument("--task_type", default="zeroshot", type=str)
    parser.add_argument("--emb_type", default="mwe", type=str)
    parser.add_argument("--duration", default=3, type=int)
    parser.add_argument("--supervisions", default="tag_artist_track", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    # runner
    parser.add_argument("--margin", default=0.4, type=float)
    parser.add_argument("--opt_type", default="SGD_Plat", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool)


    args = parser.parse_args()
    main(args)