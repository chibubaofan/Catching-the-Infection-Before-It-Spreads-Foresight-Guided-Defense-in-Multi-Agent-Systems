accelerate launch  simulation_batch.py \
    --agent_data ./data/million_villagers_1024.json \
    --album_data ./data/album_pool/{} \
    --high \
    --vlm llava-hf/llava-1.5-7b-hf \
    --clip openai/clip-vit-large-patch14 \
    --num_agents 128 \
    --num_rounds 65 \
    --max_new_tokens 128 \
    --seed 60 \
