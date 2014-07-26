abnet
=====

ABNet is a "same/different"-based loss trained neural net.


To reproduce the results in the IEEE SLT 2014 paper, you need:
 - TIMIT with the standard train/dev/test split
 - To apply `make prepare_timit dataset=PATH_TO_YOUR_TIMIT` with [the timit tools](https://github.com/SnippyHolloW/timit_tools)
 - `python align_words.py PATH_TO_TIMIT_TRAIN_FOLDER`
