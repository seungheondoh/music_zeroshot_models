import torch
import torchaudio
from torch import nn

class EmbModel(nn.Module):
    def __init__(
        self,
        audio_model: nn.Module,
        projection_ndim: int,
    ):
        super(EmbModel, self).__init__()
        self.audio_model = audio_model
        self.text_projection = nn.Linear(in_features=300, out_features=projection_ndim)

    def forward(self, waveforms, pos_repr, neg_repr):
        audio_emb = self.audio_model(waveforms) 
        pos_emb = self.text_projection(pos_repr)
        neg_emb = self.text_projection(neg_repr)
        return audio_emb, pos_emb, neg_emb
