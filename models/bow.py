import torch.nn as nn
import torch
import torch.nn.functional as F

class BoWModel(nn.Module):
    def __init__(self, vocab_features_dim, embedding_dim,  image_features_dim, output_dim):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_features_dim, embedding_dim, mode='mean')
        self.linearLayer = nn.Linear(image_features_dim + embedding_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_features, image_features):
        embeddings = self.embedding_bag(word_features)
        features = torch.cat((embeddings, image_features), 1)
        linear_out = self.linearLayer(features)
        out = self.softmax(linear_out)
        return out






