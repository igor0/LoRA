#!/bin/bash
set -e

LORA_DIM=64
SUFFIX=V
NAME=lora.$LORA_DIM.$SUFFIX
REF_FILE=e2e_ref.$NAME.txt
PRED_FILE=e2e_pred.$NAME.txt
WDIR=./trained_models/GPT2_M/e2e.$NAME
CHECKPOINT=$WDIR/model.26290.pt

# Cleanup past runs
rm -rf $WDIR
rm -f $REF_FILE $PRED_FILE

# FInetune the model
python -m torch.distributed.launch --nproc_per_node=1 --master_port=53872 src/gpt2_ft.py \
    --train_data "./data/e2e/train.jsonl" \
    --valid_data "./data/e2e/valid.jsonl" \
    --train_batch_size 10 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card "gpt2.md" \
    --init_checkpoint "./pretrained_checkpoints/gpt2-medium-pytorch_model.bin" \
    --platform "local" \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim $LORA_DIM \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_enable "FFT" \
    --label_smooth 0.1 \
    --work_dir $WDIR \
    --random_seed 110

# Run beam search
python -m torch.distributed.launch --nproc_per_node=1 --master_port=53871 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint "$CHECKPOINT" \
    --platform local \
    --lora_dim $LORA_DIM \
    --lora_alpha 32 \
    --lora_enable "FFT" \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir "$WDIR" \
    --output_file predict.jsonl

# Decode tokens to text
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file $WDIR/predict.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file "$REF_FILE" \
    --output_pred_file "$PRED_FILE"

# Measure scores
python "eval/e2e/measure_scores.py" "$REF_FILE" "$PRED_FILE" -p
