#!/bin/bash

ontonotes_path=$1
data_dir=conll_data
mkdir $data_dir

dlx() {
  wget -P $data_dir $1/$2
  tar -xvzf $data_dir/$2 -C $data_dir
  rm $data_dir/$2
}

conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz

dlx $conll_url conll-2012-scripts.v3.tar.gz
dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz

git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext
git clone https://huggingface.co/hfl/chinese-bert-wwm

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

compile_language chinese

python minimize.py $data_dir $data_dir hfl/chinese-bert-wwm

roberta_data_dir=conll_data_roberta
mkdir $roberta_data_dir
python minimize.py $data_dir $roberta_data_dir hfl/chinese-roberta-wwm-ext