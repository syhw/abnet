# bash buckeye/exps_word_length.sh MIN_SIZE_WORDS MAX_SIZE_WORDS NFRAMES
#aws s3 cp s3://bootphon_datasets/BUCKEYE_${1}-${2}_dev.joblib .
#aws s3 cp s3://bootphon_datasets/BUCKEYE_${1}-${2}_test.joblib .
outputname=output_${1}-${2}-${3}.txt 
THEANO_FLAGS="device=gpu1" python run_exp_buckeye.py --dataset-path=BUCKEYE_${1}-${2}_test.joblib --dataset-name="buckeye_${1}-${2}" --prefix-output-fname="small_coscos2_WORD_ONLY" --iterator-type=dtw --network-type=ab_net --epochs=1000 --nframes=$3 --debug-print=1 --debug-plot=0 --debug-time > ${outputname}
for name in small_coscos2_WORD_ONLY_buckeye*; do echo $name; aws s3 cp $name s3://bootphon_xpout/$name; done
aws s3 cp ${outputname} s3://bootphon_xpout/${outputname}
