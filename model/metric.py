import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()
        
def get_roc_auc(ground_truth, cos_embedding):
    roc_auc = []
    for idx, label in enumerate(tqdm(ground_truth)):
        predict = cos_embedding[idx]
        roc_auc.append(roc_auc_score(label, predict))
    return sum(roc_auc)/(len(roc_auc))

def tag_wise_roc_auc(ground_truth, cos_embedding, all_tags):
    tag_wise = {}
    for idx, tag in enumerate(tqdm(all_tags)):
        predict = cos_embedding[idx]
        label = ground_truth[idx]
        tag_wise[tag] = roc_auc_score(label, predict)
    return tag_wise

def get_eval(audio_embs, tag_embs, ground_turth, all_tags):
    ground_turth = ground_turth.detach().cpu().numpy()
    cos_embedding = cosine_similarity(audio_embs.detach().cpu().numpy(), tag_embs.detach().cpu().numpy())
    ann_roc_auc = get_roc_auc(ground_turth, cos_embedding)
    ret_roc_auc = get_roc_auc(ground_turth.T, cos_embedding.T)
    tag_wise = tag_wise_roc_auc(ground_turth.T, cos_embedding.T, all_tags)
    return ann_roc_auc, ret_roc_auc, tag_wise