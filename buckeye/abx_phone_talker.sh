#e.g. launch_abx_dirty.sh nnet.pickle npz11_test
export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/gsynnaeve/abnet:$PYTHONPATH
feats=${1%.*}
echo $1, $2, $feats
fwrd=${feats}_wrd
fspkr=${feats}_spkr
rm -rf $fwrd && rm -rf $fspkr && mkdir $fwrd && mkdir $fspkr && THEANO_FLAGS="device=gpu1" python embed_fbanks.py $1 $2 $fwrd $fspkr
python ~/IDSvsADS/npz2h5features.py $fwrd ${fwrd}.features
python ~/IDSvsADS/npz2h5features.py $fspkr ${fspkr}.features
python ~/IDSvsADS/ABX_score.py ${fwrd}.features buckeye.phone.talker.task --ncore=18 --force --dist=kl
#python ~/IDSvsADS/ABX_score.py ${fspkr}.features buckeye.talker.none.task --ncore=18 --force --dist=kl
#python ~/IDSvsADS/ABX_score.py ${fspkr}.features buckeye.talker.phone.task --ncore=18 --force --dist=kl
python ~/ABXpy/ABXpy/analyze.py buckeye.phone.talker.${fwrd}.kl.score buckeye.phone.talker.task buckeye.phone.talker.${fwrd}.kl.output
#python ~/ABXpy/ABXpy/analyze.py buckeye.talker.none.${fspkr}.kl.score buckeye.talker.none.task buckeye.talker.none.${fspkr}.kl.output
#python ~/ABXpy/ABXpy/analyze.py buckeye.talker.phone.${fspkr}.kl.score buckeye.talker.phone.task buckeye.talker.phone.${fspkr}.kl.output
python my_avg.py buckeye.phone.talker.${fwrd}.kl.output
#python my_avg.py buckeye.talker.none.${fspkr}.kl.output
#python my_avg.py buckeye.talker.phone.${fspkr}.kl.output
