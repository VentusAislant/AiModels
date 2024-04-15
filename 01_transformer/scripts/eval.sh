python transformer_src/eval/predict.py \
    --device_id 0 \
    --tokenizer_path ./data/ch-eng/tokenizer.model \
    --model_path ./checkpoint/model_reverse.bin \
    --config_path ./checkpoint/config.json \
    --max_gen_len 50