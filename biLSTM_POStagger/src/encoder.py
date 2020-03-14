import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_ud import word_to_ix,byte_to_ix,char_to_ix,tag_to_ix,training_data

torch.manual_seed(1)

LEARNING_RATE = 0.1

class sequenceRNN(nn.Module):
    """For char, byte embedding"""
    def __init__(self):
        super(sequenceRNN, self).__init__()
    
    def forward(self, x):
        pass

class contextRNN(nn.Module):
    """For word embedding"""
    def __init__(self):
        super(contextRNN, self).__init__()
    
    def forward(self, x):
        pass

if __name__ == "__main__":
    model = contextRNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)