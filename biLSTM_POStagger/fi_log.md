## fi, w+c
epoch: 17, loss: 433.7131
epoch: 18, loss: 376.2107
epoch: 19, loss: 373.7070
epoch: 20, loss: 320.4751
epoch: 20, loss: 320.4751, train acc: 99.07%, dev acc: 95.19%
Checkpoint saved: /content/drive/My Drive/Colab Notebooks/biLSTM/data/EP20_fi_b+c.model
test acc: 95.42%
Model state: /content/drive/My Drive/Colab Notebooks/biLSTM/data/fi_b+c.model

## fi, w
USE_WORD_EMB: True
USE_BYTE_EMB: False
USE_CHAR_EMB: False
epoch: 1, loss: 11666.4645
epoch: 2, loss: 8133.7827
epoch: 3, loss: 6444.1161
epoch: 4, loss: 5185.1004
epoch: 5, loss: 4177.9214
epoch: 5, loss: 4177.9214, train acc: 86.85%, dev acc: 79.94%
epoch: 6, loss: 3325.5686
epoch: 7, loss: 2632.5182
epoch: 8, loss: 2063.3432
epoch: 9, loss: 1589.9042
epoch: 10, loss: 1198.3674
epoch: 10, loss: 1198.3674, train acc: 94.95%, dev acc: 80.45%
epoch: 11, loss: 889.5190
epoch: 12, loss: 704.0602
epoch: 13, loss: 534.0274
epoch: 14, loss: 500.5439
...
epoch: 20, train_loss: 211.3059, train acc: 99.41%, dev acc: 81.04%
test acc: 81.94%

## fi, w+c
USE_WORD_EMB: True
USE_BYTE_EMB: False
USE_CHAR_EMB: True
epoch: 1, loss: 8259.9018
epoch: 2, loss: 4051.1006
epoch: 3, loss: 2696.8721
epoch: 4, loss: 1874.7735
epoch: 5, loss: 1330.3022
epoch: 5, loss: 1330.3022, train acc: 95.46%, dev acc: 91.13%
epoch: 6, loss: 945.2071
epoch: 7, loss: 718.2338
epoch: 8, loss: 492.4296
epoch: 9, loss: 341.3513
epoch: 10, loss: 271.5015
epoch: 10, loss: 271.5015, train acc: 98.58%, dev acc: 92.41%
epoch: 11, loss: 265.4306
epoch: 12, loss: 186.3047
epoch: 13, loss: 118.5519
epoch: 14, loss: 78.8088
epoch: 15, loss: 54.0075
epoch: 15, loss: 54.0075, train acc: 99.75%, dev acc: 93.24%
epoch: 16, loss: 40.7548
epoch: 17, loss: 36.1713
epoch: 18, loss: 29.7281
epoch: 19, loss: 27.8106
epoch: 20, loss: 26.0480
epoch: 20, loss: 26.0480, train acc: 99.87%, dev acc: 93.27%
test acc: 93.19%
Model state: /content/drive/My Drive/Colab Notebooks/biLSTM/data/fi_w+c.model

## fi, aux loss, b+c
USE_WORD_EMB: False
USE_BYTE_EMB: True
USE_CHAR_EMB: True
epoch: 1, loss: 12742.2019
epoch: 2, loss: 6886.6389
epoch: 3, loss: 5143.0620
epoch: 4, loss: 4208.3356
epoch: 5, loss: 3485.5309
epoch: 5, loss: 3485.5309, train acc: 93.70%, dev acc: 91.86%
epoch: 6, loss: 2956.5958
epoch: 7, loss: 2550.4592
epoch: 8, loss: 2172.4990
epoch: 9, loss: 1880.9984
epoch: 10, loss: 1642.6161
epoch: 10, loss: 1642.6161, train acc: 96.31%, dev acc: 93.37%
epoch: 11, loss: 1422.7960
epoch: 12, loss: 1285.7237
epoch: 13, loss: 1094.6254
epoch: 14, loss: 967.0086
epoch: 15, loss: 992.8003
epoch: 15, loss: 992.8003, train acc: 97.20%, dev acc: 93.94%
epoch: 16, loss: 847.3595
epoch: 17, loss: 701.1475
epoch: 18, loss: 562.4180
epoch: 19, loss: 552.1418
epoch: 20, loss: 426.4361
epoch: 20, loss: 426.4361, train acc: 98.70%, dev acc: 94.90%
test acc: 95.07%
Model state: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/fi_auxloss_b+c.model

## fi, aux loss, w+c

USE_WORD_EMB: True
USE_BYTE_EMB: False
USE_CHAR_EMB: True
epoch: 1, loss: 12377.7438
epoch: 2, loss: 6260.4533
epoch: 3, loss: 4178.0085
epoch: 4, loss: 2873.3172
epoch: 5, loss: 1974.2397
epoch: 5, loss: 1974.2397, train acc: 95.92%, dev acc: 91.53%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP5_fi_auxloss_w+c.model
epoch: 6, loss: 1365.2679
epoch: 7, loss: 949.9546
epoch: 8, loss: 604.0992
epoch: 9, loss: 403.1922
epoch: 10, loss: 318.9355
epoch: 10, loss: 318.9355, train acc: 98.82%, dev acc: 92.60%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP10_fi_auxloss_w+c.model
epoch: 11, loss: 278.3418
epoch: 12, loss: 204.1368
epoch: 13, loss: 132.7047
epoch: 14, loss: 111.7421
epoch: 15, loss: 78.0835
epoch: 15, loss: 78.0835, train acc: 99.74%, dev acc: 93.11%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP15_fi_auxloss_w+c.model
epoch: 16, loss: 58.3745
epoch: 17, loss: 44.7913
epoch: 18, loss: 39.3213
epoch: 19, loss: 34.1623
epoch: 20, loss: 30.7441
epoch: 20, loss: 30.7441, train acc: 99.95%, dev acc: 93.51%
Checkpoint saved: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/EP20_fi_auxloss_w+c.model
test acc: 93.97%
Model state: /Users/Karoteeni/coooode/MLpractices/biLSTM_POStagger/data/fi_auxloss_w+c.model