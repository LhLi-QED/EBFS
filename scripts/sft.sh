set -x

deepspeed src/train_sft.py \
   --max_len 32768 \
   --dataset data/sft.json \
   --apply_chat_template \
   --input_key input \
   --output_key output \
   --max_samples 10000000 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --lr_warmup_ratio 0.2 \
   --pretrain models/DeepSeek-Prover-V2-7B \
   --save_path models/sft \
   --bf16 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --flash_attn \
   --learning_rate 1e-5 \
   --load_checkpoint \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --use_tensorboard ../logs