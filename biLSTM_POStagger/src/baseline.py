"""Reproducing the work from Plank et al. (2016)
BASELINE biLSTM tagger
References: 
1. https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html 
2. https://github.com/FraLotito/pytorch-partofspeech-tagger/blob/master/post.py """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy

from data_ud import word_to_ix,byte_to_ix,char_to_ix,tag_to_ix,training_data

torch.manual_seed(1)

#--- hyperparameters ---
USE_WORD_EMB = False
USE_BYTE_EMB = False
USE_CHAR_EMB = True 

WORD_EMB_DIM = 128
BYTE_EMB_DIM = 100
CHAR_EMB_DIM = 100
MAX_SENT_LEN = 200
N_EPOCHS = 20
LEARNING_RATE = 0.1
REPORT_EVERY = 1
HIDDEN_DIM = 100

def get_word_tensor(seq, to_ix=word_to_ix, use=USE_WORD_EMB):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def get_char_tensor(seq, to_ix=char_to_ix, use=USE_CHAR_EMB):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [ [to_ix[c] for c in w] for w in seq ]
    idxs = [ torch.tensor(w, dtype=torch.long) for w in idxs ]
    return idxs

def get_byte_tensor(seq, to_ix=byte_to_ix, use=USE_BYTE_EMB):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    # idxs = [ [torch.tensor(to_ix[c],dtype=torch.long) for c in w] for w in seq ]
    idxs = [ [ to_ix[b] for c in w for b in list(c.encode())] for w in seq]
    idxs = [ Variable(torch.tensor(w, dtype=torch.long)) for w in idxs ]
    return idxs
        
class LSTMTagger(nn.Module):
    def __init__(self,hidden_dim,word_vocab_size,char_vocab_size,\
                byte_vocab_size,tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes concatenated embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.embedding_dim = 0
        if USE_WORD_EMB:
            self.embedding_dim += WORD_EMB_DIM
            self.word_embeddings = nn.Embedding(word_vocab_size,WORD_EMB_DIM)
            self.position_emb = nn.Embedding(MAX_SENT_LEN,WORD_EMB_DIM)
        if USE_CHAR_EMB:
            self.embedding_dim += 2*CHAR_EMB_DIM
            self.char_embeddings = nn.Embedding(char_vocab_size,CHAR_EMB_DIM)
            self.lstm_char = nn.LSTM(CHAR_EMB_DIM, self.hidden_dim, bidirectional=True)
            self.hidden_char = self.init_hidden()
        if USE_BYTE_EMB:
            self.embedding_dim += 2*BYTE_EMB_DIM
            self.byte_embeddings = nn.Embedding(byte_vocab_size,BYTE_EMB_DIM)
            self.lstm_byte = nn.LSTM(BYTE_EMB_DIM, self.hidden_dim, bidirectional=True)
            self.hidden_byte = self.init_hidden()

        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Var[aX+b]= (a**2) * Var[X]
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        sigma = 0.2
        a = torch.sqrt(torch.tensor(sigma))
        return (Variable(a*torch.randn([2,1,self.hidden_dim])),
                Variable(a*torch.randn([2,1,self.hidden_dim])))

    def forward(self, sent):
        # WORD + (char or byte)
        if USE_WORD_EMB:
            word_idx = get_word_tensor(deepcopy(sent))
            word_emb = self.word_embeddings(word_idx).view(len(sent),1,WORD_EMB_DIM)
            # Add seq position information
            i = torch.arange(0, len(sent), dtype=torch.long)
            word_emb += self.position_emb(i).view(len(sent),1,WORD_EMB_DIM)
            bilstm_in = word_emb

        if USE_CHAR_EMB:
            final_char_emb = []
            char_idx = get_char_tensor(deepcopy(sent))
            for word in char_idx:
                char_embeds = self.char_embeddings(word)
                lstm_char_out, self.hidden_char = self.lstm_char(char_embeds.view(len(word), 1, CHAR_EMB_DIM), self.hidden_char)
                
                # final_char_emb.append(lstm_char_out[-1])
            # final_char_emb = torch.stack(final_char_emb)
            
            bilstm_in = lstm_char_out[-1].view(1,1,-1) # final_char_emb
                # bilstm_in = torch.cat((word_emb, final_char_emb), 2)
                # print() #BREAK POINT
            
            # TODO: byte
            # if USE_BYTE_EMB:
            #     final_byte_emb = []
            #     byte_idx = Variable(get_byte_tensor(sent))
            #     for word in byte_idx:
            #         byte_embeds = self.byte_embeddings(word)
            #         lstm_byte_out, self.hidden_byte = self.lstm_byte(byte_embeds.view(len(word), 1, BYTE_EMB_DIM), self.hidden_byte)
            #         final_byte_emb.append(lstm_byte_out[-1])
            #     final_byte_emb = torch.stack(final_byte_emb)
                
            #     bilstm_in = torch.cat((word_embeds, final_byte_emb), 2)
                
        
        # TODO: CHAR + BYTE
        if USE_CHAR_EMB and USE_BYTE_EMB:
            char_result = []
            char_idx = get_char_tensor(sent,char_to_ix)
            for word in char_idx:
                char_emb = self.char_embeddings(word)
                lstm_char_out, self.hidden_char = self.bilstm_char(char_emb.view(len(word), 1, CHAR_EMB_DIM), self.hidden)
            bilstm = 0
            # bilstm_in = 
        
        # bilstm_in = embeds.view(len(sent),1,self.embedding_dim)
        bilstm_out, self.hidden = self.bilstm(bilstm_in, self.hidden)
        tag_space = self.hidden2tag(bilstm_out.view((1,-1))) # .view(len(sent), -1))
        return tag_space # F.log_softmax(tag_space, dim=1)

def evaluate():
    pass

if __name__ == "__main__":

    model = LSTMTagger(HIDDEN_DIM,len(word_to_ix),len(char_to_ix),len(byte_to_ix),len(tag_to_ix))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # See what the scores are before training
    # with torch.no_grad():
    #     sentence = training_data[0][0]
    #     tag_scores = model(sentence)
    #     print(tag_scores)

    for epoch in range(N_EPOCHS):
        total_loss = 0
        for sentence, tags in training_data:
            
            model.zero_grad()
            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Get inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            targets = get_word_tensor(tags, tag_to_ix, use=True) #TODO:
            tag_scores = model(sentence)       
            loss = loss_function(tag_scores, targets[0].unsqueeze(0))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # TODO: report total loss every REPORT_EVERY
        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        # if ((epoch+1) % REPORT_EVERY) == 0:
        #     train_acc = evaluate(training_data,model,BATCH_SIZE,character_map,languages)
        #     dev_acc = evaluate(data['dev'],model,BATCH_SIZE,character_map,languages)
        #     print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
        #           (epoch+1, total_loss, train_acc, dev_acc))

    # TODO: TEST
    with torch.no_grad():
        sent = training_data[0][0]
        tag_scores = model(sent)
        print(tag_scores)