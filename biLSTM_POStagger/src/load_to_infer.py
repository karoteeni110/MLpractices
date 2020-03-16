from baseline import * 
from os.path import join

USE_WORD_EMB = True
USE_BYTE_EMB = False
USE_CHAR_EMB = False 
state_dict_path = join(data_path, 'fi_w.model')

def infer_evaluate(data,model):
    loss_function = nn.CrossEntropyLoss()
    micro_correct, word_count, macro_acc = 0,0,0
    total_loss = 0
    for i, (sentence, tags) in enumerate(data):
        # print(sentence, tags)
        model.hidden = model.init_hidden()
        targets = get_targets_tensor(tags)
        tag_scores = model(sentence)
        loss = loss_function(tag_scores, targets)
        total_loss += loss.item()

        tag_preds= torch.argmax(tag_scores,dim=1)
        micro_correct += torch.sum(torch.eq(tag_preds, targets)).item()
        word_count += len(targets)
        macro_acc += 1 if torch.equal(tag_preds, targets) else 0  

        if i+1 % int(0.1 * len(training_data)) == 0:
                print('Training:', i+1, '/', len(training_data) )
        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))
    return micro_correct/word_count * 100.0, macro_acc/len(data)*100.0, total_loss

model = LSTMTagger(HIDDEN_DIM,len(word_to_ix),len(char_to_ix),len(byte_to_ix),len(tag_to_ix))
model.load_state_dict(torch.load(state_dict_path))
model.eval()

if __name__ == "__main__":
    train_mi_acc, train_ma_acc, train_loss = infer_evaluate(training_data,model)
    dev_mi_acc, dev_ma_acc, dev_loss = infer_evaluate(dev_data,model)
    print('train_loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
        (train_loss, train_mi_acc, dev_mi_acc))

    test_mi_acc, test_ma_acc, _ = infer_evaluate(test_data, model)
    print('test acc: %.2f%%' % (test_mi_acc))