import torch
import torch.nn as nn

from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, vocab_size, image_size, hidden_size, output_size, num_layers=1, dropout_prob=0, cuda_enabled=False):
        super(RNNModel, self).__init__()
        # init properties
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cuda_enabled = cuda_enabled
        # init network structure
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=vocab_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.layer_transform = nn.Linear(image_size + hidden_size, output_size)
        #self.softmax = nn.Softmax(dim=1)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def _initialize_gru_states(self, batch_size):
        var = Variable(torch.normal(torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.ones(self.num_layers, batch_size, self.hidden_size) * .01))
        if self.cuda_enabled:
            var = var.cuda()
        return var

    def forward(self, questions, images, question_lengths, gru_states=None):
        batch_size = questions.size(0)
        # encode input questions
        gru_states = gru_states if gru_states else self._initialize_gru_states(batch_size)
        # embed questions
        embeddings = self.embedding(questions)
        # pass everything to GRU
        encodings, gru_states = self.gru(embeddings, gru_states)
        # get last states by selecting state at question's last non-padding index
        # question_lengths = question_lengths.view(batch_size,1).expand(batch_size, self.hidden_size)

        # Nobody understands whats happening in this line but it works for some reason thanks to google (just skip it)
        last_states = encodings.gather(1, question_lengths.view(-1, 1, 1).expand(encodings.size(0), 1, encodings.size(2))).view(batch_size, self.hidden_size)
        #last_states = encodings.index_select(0, question_lengths)
        # predict answers
        question_vectors = torch.cat((last_states, images), 1)
        answers = self.softmax(self.layer_transform(question_vectors))
        return answers
