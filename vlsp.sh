export PYTHONPATH="$PWD"
DATA_DIR="data/vlsp_processed"
BERT_MODEL="vinai/phobert-base"

BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=256
MAXNORM=1.0
BS=5

OUTPUT_DIR="log/train_logs/vlsp2018/vlsp2018_reproduce_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_bsz${BS}_hard_span_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}_newtrunc_debug"
mkdir -p $OUTPUT_DIR
python -W ignore trainer_vlsp.py \
--data_dir $DATA_DIR \
--bert_model $BERT_MODEL \
--max_length $MAXLEN \
--gpus="1" \
--batch_size $BS \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--val_check_interval 0.5 \
--accumulate_grad_batches 2 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $MRC_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs 20 \
--span_loss_candidates "pred_and_gold" \
--weight_span $SPAN_WEIGHT \
--warmup_steps $WARMUP \
--max_length $MAXLEN \
--gradient_clip_val $MAXNORM
