# text-classification-revise
Including chinese contents encoding, RNN with built-in LSTM, RNN with LSTM from scrach and R-CNN three models 

Raw datasets were not uploaded because of the size. So find some from the Internet. Only then can you run the code.

## Execution
* Download datasets
* Run dataPreProcess/preProcess.py to generate 5 files:
  * cnews.train.seg.txt
  * cnews.val.seg.txt
  * cnews.test.seg.txt
  * cnews.vocab.txt
  * cnews.category.txt
* Run dataPreProcess/createEncodedDataset.py for testing 

After the operations above, all raw data can be encoded with numbers as the input & output for the three models.
