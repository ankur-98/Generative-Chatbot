conda activate generative_chatbot

python src/test.py \
--model_name_or_path "./ckpt/state_tracking/opt_125" \
--model_max_length 1024 \
--penalty_alpha 0.5 \
--repetition_penalty 1.1 \
--top_k 50 \
--temperature 0.7 \
--num_beams 4 \
--num_beam_groups 2 \
--diversity_penalty 0.3

# python src/test.py \
# --model_name_or_path "./ckpt/opt_350" \
# --model_max_length 512 \
# --penalty_alpha 0.5 \
# --repetition_penalty 1.2 \
# --top_k 30 \
# --temperature 0.9 \
# --num_beams 4 \
# --num_beam_groups 2 \
# --diversity_penalty 0.9 

python src/test.py \
--model_name_or_path "./ckpt/state_tracking/roberta-base" \
--model_max_length 512 \
--penalty_alpha 0.5 \
--repetition_penalty 1.2 \
--top_k 30 \
--temperature 0.9 \
--num_beams 4 \
--num_beam_groups 2 \
--diversity_penalty 0.9
