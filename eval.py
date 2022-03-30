import json
import os
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def retrieval_task(tag_audio_label_table, tag_audio_predict_table):
    auc_retrieval = []
    for tag in tqdm(tag_audio_predict_table.columns):
        label = tag_audio_label_table[tag]
        predict = tag_audio_predict_table[tag]
        auc_retrieval.append(roc_auc_score(label, predict))
    return sum(auc_retrieval)/(len(auc_retrieval))

def annotation_task(tag_audio_label_table, tag_audio_predict_table):
    auc_annotation = []
    for track_key in tqdm(tag_audio_predict_table.index):
        label = tag_audio_label_table.loc[track_key]
        predict = tag_audio_predict_table.loc[track_key]
        auc_annotation.append(roc_auc_score(label, predict))
    return sum(auc_annotation)/(len(auc_annotation))

def zeroshot_split(args):
    fl = pd.read_csv(os.path.join(args.data_dir, "msd", "annotation", args.task_type, "tag_binary.csv"), index_col=0)
    tag_split = json.load(open(os.path.join(args.data_dir, "msd", "annotation", args.task_type, "tag_split.json"), "r"))
    seen_tag = tag_split['seen_tag']
    unseen_tag = tag_split['unseen_tag']
    track_split = json.load(open(os.path.join(args.data_dir, "msd", "annotation", args.task_type, "track_split.json"), "r"))
    train_track = track_split['train_track']
    valid_track = track_split['valid_track']
    test_track = track_split['test_track']
    return fl, unseen_tag, test_track

def get_predict(args, fl):
    inference_path = os.path.join(args.data_dir, f"msd/joint_vec/{args.task_type}/{args.emb_type}/{args.backbone}/{args.supervisions}")
    embs = []
    for msdid in tqdm(fl.index):
        emb = torch.load(os.path.join(inference_path, f"audio/{msdid}.pt"), map_location="cpu")
        embs.append(emb)
    audio_emb = torch.stack(embs)
    tag_emb = torch.load(os.path.join(inference_path, f"tag_emb.pt"), map_location="cpu")
    tag_emb = tag_emb.squeeze(0)
    cos_embedding = cosine_similarity(audio_emb.numpy(), tag_emb.numpy())
    predict = pd.DataFrame(cos_embedding, index=fl.index, columns=fl.columns)
    return predict

def main(args) -> None:
    save_path = f"exp/{args.task_type}/{args.emb_type}/{args.backbone}/{args.supervisions}/"
    fl, unseen_tag, test_track = zeroshot_split(args)
    predict = get_predict(args, fl)
    gt_anno = fl.loc[test_track]
    pd_anno = predict.loc[test_track]
    gt_ret = fl[unseen_tag]
    pd_ret = predict[unseen_tag]
    results = {
        "annotation" : annotation_task(gt_anno, pd_anno),
        "retrieval" : retrieval_task(gt_ret, pd_ret)
    }
    with open(Path(save_path, "results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", default="CNN1D", type=str)
    parser.add_argument("--tid", default="debug", type=str)
    # pipeline
    parser.add_argument("--data_dir", default="../dataset", type=str)
    parser.add_argument("--msd_dir", default="../../media/chopin21/msd_resample", type=str)
    parser.add_argument("--task_type", default="zeroshot", type=str)
    parser.add_argument("--emb_type", default="Wiki_AugMC", type=str)
    parser.add_argument("--duration", default=3, type=int)
    parser.add_argument("--supervisions", default="tag_artist_track", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
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
    parser.add_argument("--reproduce", default=True, type=str2bool)


    args = parser.parse_args()
    main(args)