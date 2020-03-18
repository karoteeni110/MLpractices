"""Reproducing the work from Plank et al. (2016)
biLSTM tagger with auxilary loss
References: 
1. https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html 
2. https://github.com/FraLotito/pytorch-partofspeech-tagger/blob/master/post.py """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from random import shuffle

from data_ud import word_to_ix,byte_to_ix,char_to_ix,tag_to_ix,freq_to_ix,tag_to_freq,\
                    training_data,dev_data,test_data
from path import data_path

torch.manual_seed(1)

#--- hyperparameters ---
save_modelname = 'sv_auxloss_w+c.model'
USE_WORD_EMB = True
USE_BYTE_EMB = False
USE_CHAR_EMB = True 

WORD_EMB_DIM = 128
BYTE_EMB_DIM = 100
CHAR_EMB_DIM = 100
MAX_SENT_LEN = 300
N_EPOCHS = 20
LEARNING_RATE = 0.1
REPORT_EVERY = 5
HIDDEN_DIM = 100

def get_freq_targets_tensor(seq, to_ix=freq_to_ix):
    idxs = [to_ix[tag_to_freq[t]] if tag_to_freq[t] in to_ix else to_ix['#UNK#'] for t in seq ]
    return torch.tensor(idxs, dtype=torch.long)

def get_targets_tensor(seq, to_ix=tag_to_ix):
    idxs = [to_ix[t] if t in to_ix else to_ix['#UNK#'] for t in seq ]
    return torch.tensor(idxs, dtype=torch.long)

def get_word_tensor(seq, to_ix=word_to_ix, use=USE_WORD_EMB):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [to_ix[w.lower()] if w.lower() in to_ix else to_ix['#UNK#'] for w in seq ]
    return torch.tensor(idxs, dtype=torch.long)

def get_char_tensor(seq, to_ix=char_to_ix, use=USE_CHAR_EMB):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    idxs = [ [to_ix[c.lower()] if c.lower() in to_ix else to_ix['#UNK#'] for c in w ] for w in seq ]
    idxs = [ torch.tensor(w, dtype =torch.long) for w in idxs ]
    return idxs

def get_byte_tensor(seq, to_ix=byte_to_ix, use=USE_BYTE_EMB):
    if not use:
        return torch.LongTensor([]).repeat(len(seq),0)
    # idxs = [ [torch.tensor(to_ix[c],dtype=torch.long) for c in w] for w in seq ]
    idxs = [ [ to_ix[b] if b in to_ix else to_ix['#UNK#'] for c in w for b in list(c.encode())] for w in seq]
    idxs = [ torch.tensor(w, dtype=torch.long) for w in idxs ]
    return idxs
        
class LSTMTagger(nn.Module):
    def __init__(self,hidden_dim,word_vocab_size,char_vocab_size,\
                byte_vocab_size,tagset_size, freqclass_size):
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

        # The linear layer that maps from hidden state space to log frequency label space
        self.hidden2freq = nn.Linear(hidden_dim*2, freqclass_size)

    def init_hidden(self):
        # Var[aX+b]= (a**2) * Var[X]
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        sigma = 0.2
        a = torch.sqrt(torch.tensor(sigma))
        return (a*torch.randn([2,1,self.hidden_dim]),
                a*torch.randn([2,1,self.hidden_dim]))

    def forward(self, sent):
        # WORD + (char or byte)
        if USE_WORD_EMB:
            word_idx = get_word_tensor(sent)
            word_emb = self.word_embeddings(word_idx).view(len(sent),1,WORD_EMB_DIM)
            # Add seq position information
            i = torch.arange(0, len(sent), dtype=torch.long)
            word_emb += self.position_emb(i).view(len(sent),1,WORD_EMB_DIM)
            bilstm_in = word_emb

        if USE_CHAR_EMB:
            final_char_emb = []
            char_idx = get_char_tensor(sent)
            for word in char_idx:
                char_embeds = self.char_embeddings(word)
                self.hidden_char = self.init_hidden()
                lstm_char_out, self.hidden_char = self.lstm_char(char_embeds.view(len(word), 1, CHAR_EMB_DIM), self.hidden_char)
                final_char_emb.append(lstm_char_out[-1])
            final_char_emb = torch.stack(final_char_emb)
            bilstm_in = final_char_emb

        if USE_BYTE_EMB:
            final_byte_emb = []
            byte_idx = get_byte_tensor(sent)
            for word in byte_idx:
                byte_embeds = self.byte_embeddings(word)
                self.hidden_byte = self.init_hidden()
                lstm_byte_out, self.hidden_byte = self.lstm_byte(byte_embeds.view(len(word), 1, BYTE_EMB_DIM), self.hidden_byte)
                final_byte_emb.append(lstm_byte_out[-1])
            final_byte_emb = torch.stack(final_byte_emb)

        if USE_CHAR_EMB and USE_BYTE_EMB: 
            bilstm_in = torch.cat((final_char_emb, final_byte_emb), 2)
        if USE_WORD_EMB and USE_CHAR_EMB:
            bilstm_in = torch.cat((word_emb, final_char_emb), 2)
        
        bilstm_out, self.hidden = self.bilstm(bilstm_in, self.hidden)
        tag_space = self.hidden2tag(bilstm_out.view(len(sent), -1))
        freq_space = self.hidden2freq(bilstm_out.view(len(sent), -1))
        return tag_space, freq_space # F.log_softmax(tag_space, dim=1)

def evaluate(data,model):
    with torch.no_grad():
        micro_correct, word_count, macro_acc = 0,0,0
        for sentence, tags in data:
            # print(sentence, tags)
            model.hidden = model.init_hidden()
            
            tag_targets = get_targets_tensor(tags)
            freq_targets = get_freq_targets_tensor(tags)
            tag_scores, freq_scores = model(sentence)        

            tag_preds = torch.argmax(tag_scores,dim=1)
            
            micro_correct += torch.sum(torch.eq(tag_preds, tag_targets)).item()
            word_count += len(tag_targets)
            macro_acc += 1 if torch.equal(tag_preds, tag_targets) else 0
        
    return micro_correct/word_count * 100.0, macro_acc/len(data)*100.0

if __name__ == "__main__":
    print('USE_WORD_EMB:', USE_WORD_EMB)
    print('USE_BYTE_EMB:', USE_BYTE_EMB)
    print('USE_CHAR_EMB:', USE_CHAR_EMB)

    model = LSTMTagger(HIDDEN_DIM,len(word_to_ix),len(char_to_ix),len(byte_to_ix),len(tag_to_ix), len(freq_to_ix))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # See what the scores are before training
    # with torch.no_grad():
    #     sentence = training_data[0][0]
    #     tag_scores = model(sentence)
    #     print(tag_scores)

    for epoch in range(N_EPOCHS):
        total_loss = 0
        shuffle(training_data)

        for i, (sentence, tags) in enumerate(training_data):
            model.zero_grad()
            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Get inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            targets = get_targets_tensor(tags)
            freq_targets = get_freq_targets_tensor(tags)

            tag_scores, freq_scores = model(sentence)

            loss1 = loss_function(tag_scores, targets)
            loss2 = loss_function(freq_scores, freq_targets)
            loss = loss1 + loss2
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # if i+1 % int(0.1 * len(training_data)) == 0:
            #     print('Training:', i+1, '/', len(training_data) )
        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        if ((epoch+1) % REPORT_EVERY) == 0:
            train_mi_acc, train_ma_acc = evaluate(training_data,model)
            dev_mi_acc, dev_ma_acc = evaluate(dev_data,model)
            print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
                  (epoch+1, total_loss, train_mi_acc, dev_mi_acc))

            checkpoint_fpath= data_path + '/EP%d_%s' % (epoch+1,save_modelname)
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_fpath)
            print('Checkpoint saved:', checkpoint_fpath)

    test_acc, _ = evaluate(test_data, model)
    print('test acc: %.2f%%' % (test_acc))
    if save_modelname != '':
        p = data_path + '/%s' % save_modelname
        torch.save(model.state_dict(), p)
        print('Model state:', p)
    
    # with torch.no_grad():
    #     sent = training_data[0][0]
    #     tag_scores = model(sent)
    #     print(tag_scores)
    #     print(torch.argmax(tag_scores,dim=1))