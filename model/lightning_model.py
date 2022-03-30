import os
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from .metric import get_eval, TripletLoss


class ZSLRunner(LightningModule):
    def __init__(self, model, margin, lr, supervisions, opt_type, inference_path=None):
        super().__init__()
        self.model = model
        self.criterion = TripletLoss(margin=margin)
        self.lr = lr
        self.supervisions = supervisions
        self.opt_type = opt_type
        self.inference_path = inference_path

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.opt_type == "SGD_Plat":
            opt = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-6,
                nesterov=True
            )
            scheduler = ReduceLROnPlateau(
                optimizer=opt,
                factor=0.2,
                patience=5,
                verbose=True,
                mode='min'
            )
            lr_scheduler = {
                'scheduler': scheduler, 
                'interval': 'epoch', # The unit of the scheduler's step size
                'frequency': 1, # The frequency of the scheduler
                'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
                'monitor': 'val_loss' # Metric to monitor
            }
        elif self.opt_type == "Adam_cos":
            opt = Adam(
                self.model.parameters(),
                lr= self.lr
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer=opt,
                T_0=10, 
                last_epoch=-1
            )
            lr_scheduler = {
                'scheduler': scheduler, 
                'interval': 'epoch', # The unit of the scheduler's step size
                'frequency': 1, # The frequency of the scheduler
                'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
                'monitor': 'val_loss' # Metric to monitor
            }

        return [opt], [lr_scheduler]

    def shared_step(self, batch):
        item_dict = batch
        audio = item_dict['audio']
        tag_loss = torch.tensor(0, dtype=torch.float64)
        artist_loss = torch.tensor(0, dtype=torch.float64)
        track_loss = torch.tensor(0, dtype=torch.float64)

        audio_emb = self.model.audio_model(audio)
        if "tag" in self.supervisions:
            pos_tag = self.model.text_projection(item_dict["pos_tag_emb"])
            neg_tag = self.model.text_projection(item_dict["neg_tag_emb"])
            tag_loss = self.criterion(audio_emb, pos_tag, neg_tag)

        if "artist" in self.supervisions:
            pos_artist = self.model.text_projection(item_dict["pos_artist_emb"])
            neg_artist = self.model.text_projection(item_dict["neg_artist_emb"])
            artist_loss = self.criterion(audio_emb, pos_artist, neg_artist)

        if "track" in self.supervisions:
            pos_track = self.model.text_projection(item_dict["pos_track_emb"])
            neg_track = self.model.text_projection(item_dict["neg_track_emb"])
            track_loss = self.criterion(audio_emb, pos_track, neg_track)
        
        # sum of ratio == 1.0
        loss = (0.33 * tag_loss) + (0.33 * artist_loss) + (0.33 * track_loss)
        return loss, tag_loss, artist_loss, track_loss

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.shared_step(batch)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def training_step_end(self, step_output):
        return step_output

    def validation_step(self, batch, batch_idx):
        loss, tag_loss, artist_loss, track_loss = self.shared_step(batch)
        return {
            "val_loss": loss, 
            "val_tag_loss": tag_loss, 
            "val_artist_loss": artist_loss, 
            "val_track_loss": track_loss
            }

    def validation_step_end(self, step_output):
        return step_output

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_tag_loss = torch.mean(torch.stack([output["val_tag_loss"] for output in outputs]))
        val_artist_loss = torch.mean(torch.stack([output["val_artist_loss"] for output in outputs]))
        val_track_loss = torch.mean(torch.stack([output["val_track_loss"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_tag_loss": val_tag_loss,
                "val_artist_loss": val_artist_loss,
                "val_track_loss": val_track_loss
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    def test_step(self, batch, batch_idx):
        item_dict = batch
        audio = item_dict['audio'] # batch, chunk, channel, 66150
        track_ids = item_dict['track_ids']
        tags = item_dict['tags']
        binary = item_dict['binary']
        tag_embs = self.model.text_projection(item_dict['all_tag_embs'])
        audio_embs = self.model.audio_model(audio.transpose(0,1)) # chunk, channel, 66150
        audio_mean = audio_embs.mean(0, False) # chunk, dim
        os.makedirs(os.path.join(self.inference_path, "audio"), exist_ok=True)
        torch.save(audio_mean.detach().cpu(), os.path.join(self.inference_path, f"audio/{track_ids[0]}.pt"))
        torch.save(tag_embs.detach().cpu(), os.path.join(self.inference_path, "tag_emb.pt"))
        # return {
        #     "track_ids" : track_ids,
        #     # "tags" : tags,
        #     # "binary" : binary.detach().cpu(),
        #     # "audio_inference" : audio_mean.detach().cpu(), # memoery issue
        #     # "tag_embs" : tag_embs.detach().cpu(),
        # }

    def test_step_end(self, step_output):
        return step_output

    # def test_epoch_end(self, outputs):
    #     # audio_embs = torch.stack([output["audio_inference"] for output in outputs], dim=0) #all sample, dim
    #     print(outputs[0])
    #     track_ids = [output["track_ids"] for output in outputs] #all tags
    #     audio_embs = torch.stack([torch.load(f"../dataset/msd/joint_vec/CNN1D/{self.supervisions}/audio/{track_id}.pt") for track_id in track_ids], dim=0) #all sample, dim
    #     print(audio_embs.shape)
    #     binary = torch.cat([output["binary"] for output in outputs], dim=0) #all sample, tag_binary
    #     tag_embs = outputs[0]["tag_embs"] #tag_binary, dim
    #     all_tags = [tag[0] for tag in outputs[0]["tags"]] #all tags
    #     tag_embs = tag_embs.squeeze(0)
    #     ann_roc_auc, ann_pr_auc, ret_roc_auc, ret_pr_auc, tag_wise = get_eval(audio_embs, tag_embs, binary, all_tags)
    #     results = {
    #         'ann_roc_auc': ann_roc_auc, 
    #         'ann_pr_auc': ann_pr_auc,
    #         'ret_roc_auc': ret_roc_auc,
    #         'ret_pr_auc': ret_pr_auc,
    #         'tag_wise_roc_auc':tag_wise,
    #     }
    #     self.eval_results = results