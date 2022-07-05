#!/bin/bash
python3 t5_paraphrase_batch.py \
        --data_dir $1 \
        --top_k 0 \
        --top_p 0.95 \
        --output_dir ./output_file_$2/ \
        --batch_size 16 \
        --model_name_or_path 'UBC-NLP/ptsm_t5_paraphraser' \
        --file_name $2.tsv

#$2.tsv: a tsv file with three columns: ['tweet_id', 'label', 'content']