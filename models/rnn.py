import torch
import torch.nn as nn

from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, vocab_size, image_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        # init properties
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # init network structure
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.layer_transform = nn.Linear(image_size + hidden_size, output_size)
        self.softmax = nn.Softmax()

    def _initialize_gru_state(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

    def forward(self, question, image, gru_state=None):
        # encode input questions
        gru_state = gru_state if gru_state else self._initialize_gru_state()
        embeddings = self.embedding(question).view(len(question), 1, self.hidden_size)
        encodings, gru_state = self.gru(embeddings, gru_state)
        last_state = encodings[-1][0]
        # predict answers
        question_vector = torch.cat((last_state, image))
        # question_vector = torch.cat((last_state, image)).view(1, 1, len(image) + self.hidden_size)
        answers = self.softmax(self.layer_transform(question_vector).view(1, -1)) # softmax want some batches
        return answers