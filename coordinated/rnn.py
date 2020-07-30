import torch
import torch.nn as nn
import numpy as np

from coordinated import configs


class RNN(nn.Module):
    def __init__(self, flavor, input_size):
        super().__init__()
        self.hidden_size = 2
        self.input_size = input_size

        # define just rnn without output layer
        self.embed = torch.nn.Embedding(self.input_size, self.hidden_size)
        if flavor == 'lstm':
            cell = torch.nn.LSTM
        elif flavor == 'srn':
            cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "flavor".')
        self.encode = cell(input_size=self.hidden_size,
                           hidden_size=self.hidden_size,
                           batch_first=True,
                           nonlinearity='tanh',
                           bias=True,
                           num_layers=1,
                           dropout=0)

        # weight max really matters for good clustering
        self.embed.weight.data.uniform_(-configs.Training.max_init_weight, +configs.Training.max_init_weight)

        self.cuda()

    def forward(self,
                inputs: torch.cuda.LongTensor
                ) -> torch.cuda.FloatTensor:

        embedded = self.embed(inputs)
        encoded, _ = self.encode(embedded)  # returns all time steps [batch_size, context_size, hidden_size]
        last_encodings = torch.squeeze(encoded[:, -1])  # [batch_size, hidden_size]

        return last_encodings