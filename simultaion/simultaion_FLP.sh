
python  -u simulation_FLP.py \
    --output_point ../data/results/multi_persona/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_p16.csv \
    --agent_data ../data/million_villagers_1024.json \
    --attack_image  ../data/attack_image/border8.png \
    --num_attacks 4\
    --malicious_threshold 0.18 \
    --album_data ../data/album_pool/{} \
    --high \
    --vlm llava-hf/llava-1.5-7b-hf \
    --clip openai/clip-vit-large-patch14 \
    --num_agents 128 \
    --num_rounds 65 \
    --batch_size 4 \
    --max_new_tokens 128 \
    --slice_size 3  
