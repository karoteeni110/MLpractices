"""Reproducing the work from Plank et al. (2016)
BASELINE biLSTM tagger, using only word embeddings"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_ud import word_to_ix,byte_to_ix,char_to_ix,tag_to_ix,read_ud

torch.manual_seed(1)

#--- hyperparameters ---
EMBED_LEVEL = ['word','byte','char']
N_EPOCHS = 20
LEARNING_RATE = 0.1
REPORT_EVERY = 5
HIDDEN_DIM = 100

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# training_data = dict()



class LSTMTagger(nn.Module):
    def __init__(self,hidden_dim,word_vocab_size,char_vocab_size,\
                byte_vocab_size,tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size,128)
        self.char_embeddings = nn.Embedding(char_vocab_size,100)
        self.byte_embeddings = nn.Embedding(byte_vocab_size,100)
        
        # The LSTM takes concatenated embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        embedding_dim = word_vocab_size + char_vocab_size + byte_vocab_size
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Var[aX+b]= (a**2) * Var[X]
        sigma = 0.2
        a = torch.sqrt(torch.tensor(sigma))
        return (a*torch.randn([2,1,self.hidden_dim]),
                a*torch.randn([2,1,self.hidden_dim]))
        
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # (torch.zeros(2, 1, self.hidden_dim),
        #        torch.zeros(2, 1, self.hidden_dim))

    def forward(self, word_x, char_x, byte_x):
        word_embeds = self.word_embeddings(word_x)
        char_embeds = self.char_embeddings(char_x)
        byte_embeds = self.byte_embeddings(byte_x)
        x = torch.cat([word_embeds,char_embeds,byte_embeds])
        lstm_out, self.hidden = self.bilstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return tag_space # F.log_softmax(tag_space, dim=1)

if __name__ == "__main__":

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)

    for epoch in range(N_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            
            model.zero_grad()
            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Get inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)

        # "the dog ate the apple", DET NOUN VERB DET NOUN
        # tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
        print(tag_scores)