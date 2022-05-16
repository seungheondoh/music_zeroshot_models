DATADIR="../dataset"
mkdir $DATADIR
PRETRAINDIR="../dataset/pretrained"
mkdir $PRETRAINDIR
cd $PRETRAINDIR
wget https://zenodo.org/record/6395456/files/music_zeroshot_models.zip
unzip music_zeroshot_models.zip