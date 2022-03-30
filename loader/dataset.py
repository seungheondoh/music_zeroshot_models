import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class MSD_Dataset(Dataset):
    def __init__(self, data_dir, msd_dir, split, task_type, emb_type, supervisions, duration):
        self.data_dir = data_dir
        self.msd_dir = msd_dir
        self.split = split
        self.task_type = task_type
        self.emb_type = emb_type
        self.supervisions = supervisions
        self.num_chunks = 8
        self.msd_to_id = pickle.load(open(os.path.join(self.data_dir, "msd/annotation/MSD_id_to_7D_id.pkl"), 'rb'))
        self.id_to_path = pickle.load(open(os.path.join(self.data_dir, "msd/annotation/7D_id_to_path.pkl"), 'rb'))
        self.input_length = int(22050 * duration)
        self.tag_binary = pd.read_csv(os.path.join(self.data_dir, f"msd/annotation/{self.task_type}/tag_binary.csv"), index_col=0)
        self.get_track_split()
        if task_type == "zeroshot":
            self.get_artist_split()
            self.get_tag_split()
            self.get_zeroshot_split()
        elif task_type == "tagging":
            # self.get_artist_split()
            self.get_tagging_split()

        if "tag" in supervisions: 
            self.tag_emb = torch.load(os.path.join(self.data_dir,f"msd/w2v_vec/tag/{self.emb_type}.pt"))
        if "artist" in supervisions:
            self.tag_emb = torch.load(os.path.join(self.data_dir,f"msd/w2v_vec/tag/{self.emb_type}.pt"))
            self.artist_emb = torch.load(os.path.join(self.data_dir,f"msd/w2v_vec/artist/{self.emb_type}.pt"))
        if "track" in supervisions:
            self.tag_emb = torch.load(os.path.join(self.data_dir,f"msd/w2v_vec/tag/{self.emb_type}.pt"))
            self.track_emb = torch.load(os.path.join(self.data_dir,f"msd/w2v_vec/track/{self.emb_type}.pt"))

    def get_tag_split(self):
        tag_split = json.load(open(os.path.join(self.data_dir, f"msd/annotation/{self.task_type}/tag_split.json"), "r"))
        self.seen_tag = tag_split['seen_tag']
        self.unseen_tag = tag_split['unseen_tag']
        
    def get_track_split(self):
        track_split = json.load(open(os.path.join(self.data_dir, f"msd/annotation/{self.task_type}/track_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
        
    def get_artist_split(self):
        artist_split = json.load(open(os.path.join(self.data_dir, f"msd/annotation/{self.task_type}/artist_split.json"), "r"))
        self.train_artist = artist_split['train_artist']
        self.valid_artist = artist_split['valid_artist']
        self.artist_dict = json.load(open(os.path.join(self.data_dir, f"msd/annotation/{self.task_type}/artist_dict.json"), "r"))
        
    def get_zeroshot_split(self):
        if self.split == "TRAIN":
            self.fl = self.tag_binary[self.seen_tag].loc[self.train_track]
            self.unique_artist = self.train_artist
        elif self.split == "VALID":
            self.fl = self.tag_binary[self.seen_tag].loc[self.valid_track]
            self.unique_artist = self.valid_artist
        elif self.split == "TEST":
            self.fl = self.tag_binary
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
    
    def get_tagging_split(self):
        if self.split == "TRAIN":
            self.fl = self.tag_binary.loc[self.train_track]
            self.unique_artist = self.train_artist
        elif self.split == "VALID":
            self.fl = self.tag_binary.loc[self.valid_track]
            self.unique_artist = self.valid_artist
        elif self.split == "TEST":
            self.fl = self.tag_binary.loc[self.test_track]
        else:
            raise ValueError(f"Unexpected split name: {self.split}")

    def get_lexvec_split(self):
        if self.split == "TRAIN":
            self.fl = self.tag_binary[self.seen_tag].loc[self.train_track]
        elif self.split == "VALID":
            self.fl = self.tag_binary[self.seen_tag].loc[self.valid_track]
        elif self.split == "TEST":
            self.fl = self.tag_binary
        else:
            raise ValueError(f"Unexpected split name: {self.split}")

    def get_train_item(self, index):
        pos_tag_emb, neg_tag_emb, pos_artist_emb, neg_artist_emb, pos_track_emb, neg_track_emb = [], [], [], [], [] ,[]
        item = self.fl.iloc[index]
        audio_path = self.id_to_path[self.msd_to_id[item.name]]
        audio = np.load(os.path.join(self.msd_dir, audio_path.replace(".mp3",".npy")), mmap_mode='r')
        random_idx = random.randint(0, audio.shape[1]-self.input_length)
        audio = torch.from_numpy(np.array(audio[:,random_idx:random_idx+self.input_length]))
        label = torch.ones(1)

        if "tag" in self.supervisions:
            pos_tags = list(item[item != 0].index)
            sampled_pos_tags = random.choice(pos_tags)
            pos_tag_emb = self.tag_emb[sampled_pos_tags]
            pos_tag_emb = torch.from_numpy(pos_tag_emb.astype(np.float32))

            neg_tags = list(item[item == 0].index)
            sampled_neg_tags = random.choice(neg_tags)
            neg_tag_emb = self.tag_emb[sampled_neg_tags]
            neg_tag_emb = torch.from_numpy(neg_tag_emb.astype(np.float32))

        if "artist" in self.supervisions:
            pos_artist = self.artist_dict[item.name]
            pos_artist_emb = self.artist_emb[pos_artist]
            pos_artist_emb = torch.from_numpy(pos_artist_emb.astype(np.float32))

            neg_artists = list(self.unique_artist)
            neg_artists.remove(pos_artist)
            neg_artist = random.choice(neg_artists)
            neg_artist_emb = self.artist_emb[neg_artist]
            neg_artist_emb = torch.from_numpy(neg_artist_emb.astype(np.float32))

        if "track" in self.supervisions:
            pos_track_emb = self.track_emb[item.name]
            pos_track_emb = torch.from_numpy(pos_track_emb.astype(np.float32))

            neg_tracks = self.fl.index.drop(item.name)
            neg_track = random.choice(neg_tracks)
            neg_track_emb = self.track_emb[neg_track]
            neg_track_emb = torch.from_numpy(neg_track_emb.astype(np.float32))

        return {
                    "audio":audio, "label":label,
                    "pos_tag_emb": pos_tag_emb, "neg_tag_emb": neg_tag_emb, 
                    "pos_artist_emb": pos_artist_emb, "neg_artist_emb":neg_artist_emb, 
                    "pos_track_emb":pos_track_emb, "neg_track_emb":neg_track_emb,
                }

    def get_eval_item(self, index):
        item = self.fl.iloc[index]
        binary = item.values
        tags = list(self.fl.columns)
        all_tag_embs = np.stack([self.tag_emb[i] for i in tags])
        audio_path = self.id_to_path[self.msd_to_id[item.name]]
        audio = np.load(os.path.join(self.msd_dir, audio_path.replace(".mp3",".npy")), mmap_mode='r')
        audio = audio.squeeze(0)
        hop = (len(audio) - self.input_length) // self.num_chunks
        audio = np.array([audio[i * hop : i * hop + self.input_length] for i in range(self.num_chunks)]).astype('float32')
        return {
                    "audio":audio, 
                    "track_ids":item.name,
                    "binary":binary,
                    "tags": tags,
                    "all_tag_embs": all_tag_embs
                }

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)
            
    def __len__(self):
        return len(self.fl)