import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer 

def load_checkpoint(checkpoint_dir, checkpoint_files):
    state_dict = {}
    for file in checkpoint_files:
        checkpoint = torch.load(os.path.join(checkpoint_dir, file), map_location='cpu')
        state_dict.update(checkpoint)
    return state_dict

def convert_model_to_hf(checkpoint_dir, checkpoint_files, base_model_path, output_dir, tokenizer_model_path):
    # Check if output directory already exists
    if os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} already exists. Please choose a different directory to avoid overwriting.")

    # Load the model state dict from checkpoints
    state_dict = load_checkpoint(checkpoint_dir, checkpoint_files)
    
    # Load the base model and configuration
    config = AutoConfig.from_pretrained(base_model_path)
    hf_model = AutoModelForCausalLM.from_config(config)
    
    # Convert state_dict to bfloat16
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            state_dict[key] = value.to(torch.bfloat16)
    
    # Load the state dict into the model
    hf_model.load_state_dict(state_dict, strict=True)
    
    # Ensure the model is in bfloat16
    hf_model = hf_model.to(torch.bfloat16)
    
    # Save the model
    hf_model.save_pretrained(output_dir)
    
    # Save the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved successfully from {tokenizer_model_path}")
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_model_path}: {str(e)}")
        print("Attempting to load tokenizer from base model path...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print(f"Tokenizer saved successfully from {base_model_path}")
        except Exception as e:
            print(f"Error loading tokenizer from base model path: {str(e)}")
            print("Failed to load and save tokenizer.")

    print(f"Model safely saved to new directory: {output_dir}")
    print(f"Model dtype: {next(hf_model.parameters()).dtype}")

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Usage
checkpoint_dir = '/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/8b_new_era_train'
checkpoint_files = [
            "/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/8b_new_era_train/hf_model_0001_300.pt",
            "/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/8b_new_era_train/hf_model_0002_300.pt",
            "/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/8b_new_era_train/hf_model_0003_300.pt",
            "/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/8b_new_era_train/hf_model_0004_300.pt"
]
base_model_path = '/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/Meta-Llama-3.1-8B-Instruct'
output_dir = '/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/8b_sft_v1'
tokenizer_model_path = '/data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/checkpoints/llama3_2_1b_Instruct'

convert_model_to_hf(checkpoint_dir, checkpoint_files, base_model_path, output_dir, tokenizer_model_path)
