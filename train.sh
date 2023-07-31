conda activate generative_chatbot

# python src/train.py \
# --model_name_or_path "facebook/opt-125m" \
# --output_dir "./ckpt/state_tracking/opt_125" \
# --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 4 \
# --model_max_length 512 \
# --fp16 \
# --fp16_opt_level "O1" \
# --load_best_model_at_end \
# --metric_for_best_model "eval_loss" \
# --save_strategy "steps" \
# --evaluation_strategy "steps" \
# --logging_first_step \
# --logging_steps 100 \
# --save_steps 100 \
# --save_total_limit 1 \
# --overwrite_output_dir \
# --seed 0 \
# --resume_from_checkpoint "./ckpt/state_tracking/opt_125"

python src/train.py \
--model_name_or_path "facebook/opt-350m" \
--output_dir "./ckpt/state_tracking/opt_350" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--model_max_length 512 \
--fp16 \
--fp16_opt_level "O1" \
--load_best_model_at_end \
--metric_for_best_model "eval_loss" \
--save_strategy "steps" \
--evaluation_strategy "steps" \
--logging_first_step \
--logging_steps 100 \
--save_steps 100 \
--save_total_limit 1 \
--overwrite_output_dir \
--seed 0 \
--resume_from_checkpoint "./ckpt/state_tracking/opt_350"


python src/train.py \
--model_name_or_path "roberta-base" \
--output_dir "./ckpt/state_tracking/roberta-base" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--model_max_length 512 \
--fp16 \
--fp16_opt_level "O1" \
--load_best_model_at_end \
--metric_for_best_model "eval_loss" \
--save_strategy "steps" \
--evaluation_strategy "steps" \
--logging_first_step \
--logging_steps 100 \
--save_steps 100 \
--save_total_limit 1 \
--overwrite_output_dir \
--seed 0 \
--resume_from_checkpoint "./ckpt/state_tracking/roberta-base" 
