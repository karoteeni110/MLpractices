"""Reproducing the work from Plank et al. (2016)
BASELINE biLSTM tagger
Based on the"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_ud import word_to_ix,byte_to_ix,char_to_ix,tag_to_ix,training_data

torch.manual_seed(1)

#--- hyperparameters ---
USE_WORD_EMB = True
USE_BYTE_EMB = False
USE_CHAR_EMB = True 

WORD_EMB_DIM = 128
BYTE_EMB_DIM = 100
CHAR_EMB_DIM = 100
N_EPOCHS = 20
LEARNING_RATE = 0.1
REPORT_EVERY = 5
HIDDEN_DIM = 100

def get_word_tensor(seq, to_ix, use=True):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def get_char_tensor(seq, to_ix, use=True):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [to_ix[c] for w in seq for c in w]
    return torch.tensor(idxs, dtype=torch.long)

def get_byte_tensor(seq, to_ix, use=True):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [to_ix[b] for w in seq for c in w for b in list(c.encode())]
    return torch.tensor(idxs, dtype=torch.long)
        
class LSTMTagger(nn.Module):
    def __init__(self,hidden_dim,word_vocab_size,char_vocab_size,\
                byte_vocab_size,tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size,WORD_EMB_DIM)
        self.char_embeddings = nn.Embedding(char_vocab_size,CHAR_EMB_DIM)
        self.byte_embeddings = nn.Embedding(byte_vocab_size,BYTE_EMB_DIM)
        
        # The LSTM takes concatenated embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        embedding_dim = 0
        if word_vocab_size != 0:
            embedding_dim += WORD_EMB_DIM
        if char_vocab_size != 0:
            embedding_dim += CHAR_EMB_DIM
        if byte_vocab_size != 0:
            embedding_dim += BYTE_EMB_DIM
        
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Var[aX+b]= (a**2) * Var[X]
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        sigma = 0.2
        a = torch.sqrt(torch.tensor(sigma))
        return (a*torch.randn([2,1,self.hidden_dim]),
                a*torch.randn([2,1,self.hidden_dim]))

    def forward(self, word_x, char_x, byte_x, sent_len):
        # TODO: concat embeddings
        word_embeds = self.word_embeddings(word_x).view(1,-1)
        char_embeds = self.char_embeddings(char_x).mean(dim=0).view(1,-1)
        byte_embeds = self.byte_embeddings(byte_x).mean(dim=0).view(1,-1)
        embeds = []
        for emb in [word_embeds,char_embeds,byte_embeds]:
            if 0 not in emb.size():
                embeds.append(emb)
        if len(embeds) > 1:
            embeds = torch.cat(embeds,dim=1)
        else:
            embeds = embeds[0]
        lstm_out, self.hidden = self.bilstm(embeds.view(sent_len,1,-1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(sent_len, -1))
        return tag_space # F.log_softmax(tag_space, dim=1)

if __name__ == "__main__":

    model = LSTMTagger(HIDDEN_DIM,len(word_to_ix),len(char_to_ix),len(byte_to_ix),len(tag_to_ix))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    with torch.no_grad():
        sentence = training_data[0][0]
        word_inputs = get_word_tensor(sentence, word_to_ix, use=USE_WORD_EMB)
        byte_inputs = get_byte_tensor(sentence, byte_to_ix, use=USE_BYTE_EMB)
        char_inputs = get_char_tensor(sentence, char_to_ix, use=USE_CHAR_EMB)
        tag_scores = model(word_inputs,char_inputs,byte_inputs,len(sentence))
        print(tag_scores)

    for epoch in range(N_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            
            model.zero_grad()
            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Get inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            word_inputs = get_word_tensor(sentence, word_to_ix, use=USE_WORD_EMB)
            byte_inputs = get_byte_tensor(sentence, byte_to_ix, use=USE_BYTE_EMB)
            char_inputs = get_char_tensor(sentence, char_to_ix, use=USE_CHAR_EMB)
            targets = get_word_tensor(tags, tag_to_ix)

            tag_scores = model(word_inputs,char_inputs,byte_inputs,len(sentence))        
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        sent_in = training_data[0][0]
        word_inputs = get_word_tensor(sent_in, word_to_ix, use=USE_WORD_EMB)
        byte_inputs = get_byte_tensor(sent_in, byte_to_ix, use=USE_BYTE_EMB)
        char_inputs = get_char_tensor(sent_in, char_to_ix, use=USE_CHAR_EMB)
        tag_scores = model(word_inputs,byte_inputs,char_inputs,len(sent_in))

        # "the dog ate the apple", DET NOUN VERB DET NOUN
        # tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
        print(tag_scores)