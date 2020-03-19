## de, w
USE_WORD_EMB: True
USE_BYTE_EMB: False
USE_CHAR_EMB: False
epoch: 1, loss: 11322.8884
epoch: 2, loss: 5979.9929
epoch: 3, loss: 4311.3720
epoch: 4, loss: 3221.3990
epoch: 5, loss: 2456.3783
epoch: 5, loss: 2456.3783, train acc: 94.15%, dev acc: 85.33%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP4_de_w.model
epoch: 6, loss: 1854.6281
epoch: 7, loss: 1399.2387
epoch: 8, loss: 1076.4374
epoch: 9, loss: 806.8803
epoch: 10, loss: 585.4004
epoch: 10, loss: 585.4004, train acc: 98.66%, dev acc: 85.83%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP9_de_w.model
epoch: 11, loss: 433.9204
epoch: 12, loss: 322.8268
epoch: 13, loss: 237.8464
epoch: 14, loss: 184.2527
epoch: 15, loss: 144.8435
epoch: 15, loss: 144.8435, train acc: 99.85%, dev acc: 85.92%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP14_de_w.model
epoch: 16, loss: 126.6383
epoch: 17, loss: 100.7902
epoch: 18, loss: 83.8726
epoch: 19, loss: 76.8086
epoch: 20, loss: 74.5635
epoch: 20, loss: 74.5635, train acc: 99.96%, dev acc: 86.10%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP19_de_w.model
test acc: 86.62%
/content/drive/My Drive/Colab Notebooks/biLSTM/data/de_hdt-ud-train.conllu.txt
Model state: /content/drive/My Drive/Colab Notebooks/biLSTM/data/de_w.model

## de, b+c
epoch: 17, loss: 238.2588
epoch: 18, loss: 192.6160
epoch: 19, loss: 174.3819
epoch: 20, loss: 153.4098
epoch: 20, loss: 153.4098, train acc: 99.48%, dev acc: 95.09%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP20_de_b+c.model
test acc: 95.50%
Model state: /content/drive/My Drive/Colab Notebooks/biLSTM/data/de_b+c.model

## de, aux loss, w
USE_WORD_EMB: True
USE_BYTE_EMB: False
USE_CHAR_EMB: False
epoch: 1, loss: 18685.0574
epoch: 2, loss: 10217.4310
epoch: 3, loss: 7398.0578
epoch: 4, loss: 5487.5898
epoch: 5, loss: 4158.6252
epoch: 5, loss: 4158.6252, train acc: 94.90%, dev acc: 85.70%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP5_de_auxloss_w.model
epoch: 6, loss: 3178.9628
epoch: 7, loss: 2391.8952
epoch: 8, loss: 1719.6351
epoch: 9, loss: 1305.8509
epoch: 10, loss: 936.1247
epoch: 10, loss: 936.1247, train acc: 99.04%, dev acc: 86.12%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP10_de_auxloss_w.model
epoch: 11, loss: 681.5937
epoch: 12, loss: 485.3935
epoch: 13, loss: 340.9362
epoch: 14, loss: 281.4984
epoch: 15, loss: 225.0123
epoch: 15, loss: 225.0123, train acc: 99.90%, dev acc: 86.05%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP15_de_auxloss_w.model
epoch: 16, loss: 179.5755
epoch: 17, loss: 143.0634
epoch: 18, loss: 127.9787
epoch: 19, loss: 112.7418
epoch: 20, loss: 114.8832
epoch: 20, loss: 114.8832, train acc: 99.96%, dev acc: 86.47%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP20_de_auxloss_w.model
test acc: 86.85%
Model state: /content/drive/My Drive/Colab Notebooks/biLSTM/data/de_auxloss_w.model

## de auxloss w+c
USE_WORD_EMB: True
USE_BYTE_EMB: False
USE_CHAR_EMB: True
epoch: 1, loss: 12750.3698
epoch: 2, loss: 5047.1272
epoch: 3, loss: 3217.9246
epoch: 4, loss: 2192.5358
epoch: 5, loss: 1539.1644
epoch: 5, loss: 1539.1644, train acc: 98.38%, dev acc: 93.41%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP5_de_auxloss_w+c.model
epoch: 6, loss: 1063.6754
epoch: 7, loss: 728.0342
epoch: 8, loss: 490.2613
epoch: 9, loss: 338.9216
epoch: 10, loss: 227.1551
epoch: 10, loss: 227.1551, train acc: 99.85%, dev acc: 94.18%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP10_de_auxloss_w+c.model
epoch: 11, loss: 147.7043
epoch: 12, loss: 109.3319
epoch: 13, loss: 85.4473
epoch: 14, loss: 69.8866
epoch: 15, loss: 58.7786
epoch: 15, loss: 58.7786, train acc: 99.98%, dev acc: 94.28%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP15_de_auxloss_w+c.model
epoch: 16, loss: 49.5639
epoch: 17, loss: 44.4513
epoch: 18, loss: 39.6131
epoch: 19, loss: 36.6810

## de, auxloss, b+c

TRAIN: /Volumes/Valar Morghulis/ud_data/de_hdt-ud-train.conllu.txt
TEST: /Volumes/Valar Morghulis/ud_data/de_hdt-ud-test.conllu.txt
DEV: /Volumes/Valar Morghulis/ud_data/de_hdt-ud-train.conllu.txt
USE_WORD_EMB: False
USE_BYTE_EMB: True
USE_CHAR_EMB: True
epoch: 1, loss: 13716.4672
epoch: 2, loss: 5321.2905
epoch: 3, loss: 3723.9908
epoch: 4, loss: 2923.0964
epoch: 5, loss: 2378.1157
epoch: 5, loss: 2378.1157, train acc: 97.08%, dev acc: 95.80%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP5_de_auxloss_b+c.model
epoch: 6, loss: 2019.4550
epoch: 7, loss: 1666.0352
epoch: 8, loss: 1360.5251
epoch: 9, loss: 1140.8483
epoch: 10, loss: 929.4076
epoch: 10, loss: 929.4076, train acc: 98.83%, dev acc: 96.97%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP10_de_auxloss_b+c.model
epoch: 11, loss: 792.3582
epoch: 12, loss: 693.2063
epoch: 13, loss: 602.4458
epoch: 14, loss: 560.1024
epoch: 15, loss: 462.9350
epoch: 15, loss: 462.9350, train acc: 99.49%, dev acc: 97.30%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP15_de_auxloss_b+c.model
epoch: 16, loss: 392.3253
epoch: 17, loss: 361.8340
epoch: 18, loss: 312.1077
epoch: 19, loss: 225.3009
epoch: 20, loss: 168.5472
epoch: 20, loss: 168.5472, train acc: 99.83%, dev acc: 97.51%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP20_de_auxloss_b+c.model
test acc: 95.85%
Model state: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/de_auxloss_b+c.model

## de, aux loss, w+c
epoch: 17, loss: 54.9406
epoch: 18, loss: 49.8927
epoch: 19, loss: 43.3559
epoch: 20, loss: 37.2080
epoch: 20, loss: 37.2080, train acc: 100.00%, dev acc: 94.30%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP20_de_auxloss_w+c.model
test acc: 94.72%
Model state: /content/drive/My Drive/Colab Notebooks/biLSTM/data/de_auxloss_w+c.model