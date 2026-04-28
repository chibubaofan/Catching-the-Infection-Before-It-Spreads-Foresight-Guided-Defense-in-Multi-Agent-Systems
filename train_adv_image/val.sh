python -u validate.py\
    --vlm llava-hf/llava-1.5-7b-hf \
    --clip openai/clip-vit-large-patch14 \
    --root ../data/attack_image\
    --pixel  \
    --epsilon 16 \
    --valid_epoch 10 \
    --epochs 100 \
    --target -1 \