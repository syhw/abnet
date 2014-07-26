abnet
=====

ABNet is a "same/different"-based loss trained neural net.

#### Data preprocessing 
To reproduce the results in the IEEE SLT 2014 paper, you need:
 - TIMIT with the standard train/dev/test split
 - To apply `make prepare_timit dataset=PATH_TO_YOUR_TIMIT` with [the timit tools](https://github.com/SnippyHolloW/timit_tools), that will create all the needed features (Mel filterbanks), for this step, [spectral](https://github.com/mwv/spectral) is a requirement.

#### Training a (deep) ABnet 
Then you can:
 - Align words and extract their dynamic time warped paths with:
```
python align_words.py PATH_TO_TIMIT_TRAIN_FOLDER && python align_words.py PATH_TO_TIMIT_DEV_FOLDER && python align_words.py PATH_TO_TIMIT_TEST_FOLDER
```
See in this `align_words.py` for variants / size of words. This step needs [DTW_Cython](https://github.com/SnippyHolloW/DTW_Cython)
 - Train the ABnet on this DTW aligned word patterns, e.g. with:
```
THEANO_FLAGS="device=gpu0" python run_exp_AB.py --dataset-path=dtw_words_train.joblib --dataset-name="timit_dtw" --prefix-output-fname="deep_cos_cos2" --iterator-type=dtw --nframes=7 --network-type=ab_net --debug-print=0 --debug-plot=0 --debug-time
```

#### ABX evaluation
If you want to evaluate it with ABX, you need to:
 - Create a folder with `*.npz` files containing timing and features, for that copy every filterbank numpy array and stack them as needed, e.g. with:
```
for name in `find . -name "*_fbanks.npy" | grep train`; do cp $name npz7_train/`echo $name | awk -F '/' '{print $4"_"$5}'`; done
python stack_fbanks.py npz7_train/*.npy
```
 - Use your trained ABnet to make the transformation of these filterbanks into the embedded features of the ABnet:
```
mkdir deep_cos_cos2 && python embed_fbanks.py deep_cos_cos2_timit_dtw_fbank7_ab_net_adadelta.pickle PATH_TO_npz7_train deep_cos_cos2
```
 - Make an ABX compatible `*.features` HDF5 file using: `python npz2h5features.py deep_cos_cos2 deep_cos_cos2.features`
 - You can now do an ABX evaluation e.g. with:
```
python ABX_repo/ABX_score.py deep_cos_cos2.features timit_ABX_train.phone.talker.task --ncore=8 --force
python ABX_repo/collpanda.py timit_ABX_train.phone.talker.deep_cos_cos2.score timit_ABX_train.phone.talker.task timit_ABX_train.phone.talker.deep_cos_cos2.output
bash ABX_repo/avg timit_ABX_train.phone.talker.deep_cos_cos2.output
```
