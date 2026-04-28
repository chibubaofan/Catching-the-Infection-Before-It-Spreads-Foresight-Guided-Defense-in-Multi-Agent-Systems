
accelerate launch  --config_file accelerate_config.yaml optimize.py \
    --vlm llava-hf/llava-1.5-7b-hf \
    --clip openai/clip-vit-large-patch14 \
    --root ../data/attack_image\
    --pixel \
    --epsilon 16
