## Incorporating Syntax and Semantics in Coreference Resolution with Heterogeneous Graph Attention Network

### Setup

- pip -r install requirements.txt
- ```./setup_training.sh <ontonotes/path/ontonotes-release-5.0> conll_data```.
This assumes that you have access to OntoNotes 5.0. The preprocessed data will be included under ```conll_data```.

### Extract Dependency Trees
- ```wget https://nlp.stanford.edu/software/stanford-parser-4.0.0.zip``` and 
  ```!unzip stanford-parser-4.0.0.zip``` to get the stanford parser.
- ```python extract_constituency.py``` and ```python extract_gold_dependency``` will convert gold constituent
parsing trees to Stanford dependency trees, which will be under ```conll_data```.

### Build Kernels
- ```python setup.py install```. This will build kernel for extracting top spans implemented using the C++ interface
of Pytorch.


### Training
- ```python train.py <experiment>```
- Results are stored in the ```log_root``` directory.
- For getting the result of using SpanBERT-Base and SpanBERT-Large model, use 
```python train.py train_spanbert_base_hgat_dep_srl_two_way``` and ```python train.py train_spanbert_large_hgat_dep_srl_two_way```
- Finetuning a SpanBERT large model on OntoNotes requires access to a 32GB GPU, while the base model
can be trained in a 16GB GPU.