import torch
import torch.nn as nn
import string

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.ALL_LETTERS = string.ascii_letters + " .,;'"

    def char_tensor(self, letter):
        tensor = torch.zeros(1, self.input_size)
        index = self.ALL_LETTERS.find(letter)
        if index != -1:
            tensor[0][index] = 1
        return tensor

    def encode(self, node):
        if node is None:
            return torch.zeros(1, self.hidden_size)
        
        left_hidden = self.encode(node['left'])
        right_hidden = self.encode(node['right'])

        input_tensor = self.char_tensor(node['value'])
        combined = torch.cat((input_tensor, left_hidden + right_hidden), 1)
        hidden = self.i2h(combined)
        hidden = torch.tanh(hidden)  # Optional non-linearity
        return hidden

    def forward(self, tree):
        hidden = self.encode(tree)
        output = self.i2o(hidden)
        output = self.softmax(output)
        return output
