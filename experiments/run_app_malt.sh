#!/bin/bash
cd ..
cd app-malt

# Define common parameters
NUM_QUERIES=1
BENCHMARK_PATH="data/sampled_1_benchmark_malt.jsonl"
PROMPT_TYPE="few_shot_semantic"  # Define prompt_type
# PROMPT_TYPE="few_shot_basic"  # Define prompt_type
# PROMPT_TYPE="zero_shot_cot"  # Define prompt_type

export HUGGINGFACE_TOKEN="[YOUR_TOKEN_HERE]"  # Set your Hugging Face token here

# Azure OpenAI configuration
export AZURE_OPENAI_ENDPOINT="https://your-gpt-4o-deployment.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-gpt-4o-deployment"
export AZURE_OPENAI_API_VERSION="YYYY-MM-DD" # Optional. Defaults to "2024-10-01".
export AZURE_OPENAI_API_KEY="API_KEY" # Not needed if Entra ID is used.

# Function to run experiment for a model
run_experiment() {
    local llm_model_type=$1
    local prompt_type=$2
    local complexity=$3
    local output_file=$4
    
    # Create agent-specific output directory
    local agent_output_dir="logs/${llm_model_type}_${prompt_type}"
    
    echo "Running experiment for $llm_model_type..."
    
    # Clear GPU cache before running
    python -c "import torch; torch.cuda.empty_cache()" 
    
    python main.py \
        --llm_model_type "$llm_model_type" \
        --prompt_type "$PROMPT_TYPE" \
        --num_queries $NUM_QUERIES \
        --complexity_level $complexity \
        --output_dir "$agent_output_dir" \
        --output_file "$output_file" \
        --dynamic_benchmark_path "$BENCHMARK_PATH" \
        --regenerate_query
    }

# Define models and their configurations
declare -A model_configs=(
    ["Qwen2.5-72B-Instruct"]="level1 level2 level3:qwen_fewshot_semantic_50.jsonl"
)

# Define the desired order of execution
model_order=("Qwen2.5-72B-Instruct")

# Run experiments in specified order
for model in "${model_order[@]}"; do
    echo "==============================================="
    echo "Starting experiment with model: $model, prompt type: $PROMPT_TYPE"
    
    IFS=':' read -r complexity output_file <<< "${model_configs[$model]}"
    run_experiment "$model" "$PROMPT_TYPE" "$complexity" "$output_file"
    
    echo "Finished experiment with model: $model, prompt type: $PROMPT_TYPE"
    echo "==============================================="
    # Add a small delay between experiments
    sleep 5
done

echo "All experiments completed!" 