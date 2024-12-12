#!/bin/bash

if [ ! -d "/tmp/wmt14_en_de" ]; then
    echo "Downloading dataset"
    wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/wmt_data/databin_wmt14_en_de.tar.gz -P /tmp
    tar -zxvf /tmp/databin_wmt14_en_de.tar.gz -C /tmp && rm /tmp/databin_wmt14_en_de.tar.gz
fi

if [ -d "checkpoints" ]; then
    rm -r checkpoints
fi

if [ -d ".cache" ]; then
    rm -r .cache
fi

# LightSeq2 Transformer
lightseq-train /tmp/wmt14_en_de/ \
    --task translation \
    --arch ls_transformer_wmt_en_de_big_t2t \
    --encoder-layers $1 \
    --decoder-layers $1 \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $2 \
    --log-format simple --log-interval 10 \
    --distributed-world-size 1 \
    --stop-time-hours 0.05 \
    --memory-efficient-fp16
