# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

# Machine Translation
## Prepare the dataset
Go to https://github.com/pytorch/fairseq/tree/master/examples/translation to prepare the dataset for IWSLT. 

## Training with standard pipeline
python fairseq_cli/train.py  data-bin/iwslt14.tokenized.de-en               --arch transformer_iwslt_de_en             --share-decoder-input-output-embed             --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0             --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000              --dropout 0.3 --weight-decay 0.0001              --c
riterion label_smoothed_cross_entropy --label-smoothing 0.1             --max-tokens 4096  --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'             --eval-bleu-detok moses     --eval-bleu-remove-bpe             --no-progress-bar --keep-last-epochs 5             --log-interval 50             --save-dir iwslt_result/standard  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
       --max-epoch 55             | tee -a iwslt_result/standard/train_log.txt

## Training with our quantized model
fairseq_cli/train.py  data-bin/iwslt14.tokenized.de-en               --arch qtransformer_iwslt_de_en             --share-decoder-input-output-embed             --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0             --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000              --dropout 0.3 --weight-decay 0.0001              --
criterion label_smoothed_cross_entropy --label-smoothing 0.1             --max-tokens 4096  --eval-bleu --eval-bleu-args  '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'             --eval-bleu-detok moses     --eval-bleu-remove-bpe             --no-progress-bar --keep-last-epochs 5 --max-epoch 55             --log-interval 50             --save-dir iwslt_result/quantized_transformer_4bit  --best-checkpoint-metric bleu --ma
ximize-best-checkpoint-metric             --quantize-bit 8             | tee -a iwslt_result/quantized_transformer_8bit/train_log.txt


## code used for quantized training are mainly contained:
fairseq/modules/transformer_layer_quantize.py
fairseq/modules/multihead_attention_quantize.py
fairseq/modules/support_quantize_layers.py

If you want to profile the memory usage:
fairseq/criterions/label_smoothed_cross_entropy.py 

# Language Model
## Prepare the dataset
Go to https://github.com/pytorch/fairseq/tree/master/examples/language_model to prepare the dataset for wikitext103

## Training with standard pipeline 
python fairseq_cli/train.py   data-bin/wikitext-103   --task language_modeling   --save-dir wikitext_result/standard   --arch transformer_lm --share-decoder-input-output-embed   --dropout 0.1   --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0   --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --w
armup-init-lr 1e-07   --tokens-per-sample 512 --sample-break-mode none   --max-tokens 4096 --update-freq 16   --max-update 50000

Note that the GPU memory is 24Gb to train --max-tokens 4096

## Training with our quantized model
python fairseq_cli/train.py     data-bin/wikitext-103     --task language_modeling     --save-dir wikitext_result/quantized_transformer_8bit     --arch qtransformer_lm --share-decoder-input-output-embed     --dropout 0.1     --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0     --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07     --tokens-per-sample 512 --sample-break-mode none     --max-tokens 4096 --update-freq 16     --max-update 50000 

## code used for quantized training are mainly contained: 
fairseq/modules/transformer_layer_quantize.py
fairseq/modules/multihead_attention_quantize.py
fairseq/modules/support_quantize_layers.py
fairseq/models/transformer_lm_quantize.py

## Issues
I did not see any memory saving for this model! Not sure if I made any error in the code.