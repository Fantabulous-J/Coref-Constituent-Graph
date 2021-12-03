#!/bin/bash

ontonotes_path=$1
data_dir=conll_data
mkdir $data_dir

dlx() {
  wget -P $data_dir $1/$2
  tar -xvzf $data_dir/$2 -C $data_dir
  rm $data_dir/$2
}

download_spanbert() {
  model=$1
  model_dir=$2
  wget -P $model_dir https://dl.fbaipublicfiles.com/fairseq/models/$model.tar.gz
  mkdir $model_dir
  tar xvfz $model_dir/$model.tar.gz -C $model_dir
  rm $model_dir/$model.tar.gz
}

conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz

dlx $conll_url conll-2012-scripts.v3.tar.gz
dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz

download_spanbert spanbert_hf spanbert_large
download_spanbert spanbert_hf_base spanbert_base

bash $data_dir/conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data $data_dir/conll-2012

function compile_partition() {
  rm -f $2.$5.$3$4
  cat $data_dir/conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >>$data_dir/$2.$5.$3$4
}

function compile_language() {
  compile_partition development dev v4 _gold_conll $1
  compile_partition train train v4 _gold_conll $1
  compile_partition test test v4 _gold_conll $1
}

compile_language english

python minimize.py $data_dir $data_dir

python extract_constituency.py
