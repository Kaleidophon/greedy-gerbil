import torch.nn as nn
import torch

class BoWModel(nn.Module):
    def __init__(self, vocab_features_dim, embedding_dim,  image_features_dim, output_dim, dropout_prob=0):
        super().__init__()
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(vocab_features_dim + 1, embedding_dim, padding_idx=vocab_features_dim)
        self.linearLayer = nn.Linear(image_features_dim + embedding_dim, output_dim)
        
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
            self.dropout_enabled = True
        else:
            self.dropout_enabled = False
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_features, image_features):
        emb = self.embedding(word_features)
        bow = torch.sum(emb, 1)
        if self.dropout_prob > 0 and self.dropout_enabled:
            bow = self.dropout(bow)
        features = torch.cat((bow, image_features), 1)
        linear_out = self.linearLayer(features)
        out = self.softmax(linear_out)
        return out






