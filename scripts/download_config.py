# usage: python scripts/download_config.py --model_name allenai/OLMoE-1B-7B-0924 --output_dir config/models

import argparse
import os

from transformers import AutoConfig, AutoTokenizer

def download_config(model_name, output_dir):
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    # make model_type as model_type + "_coe" in config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as f:
        config_json = f.read()
    with open(config_path, "w") as f:
        f.write(config_json.replace(f'"{config.model_type}"', f'"{config.model_type}_coe"'))
        
    print(f"Config and tokenizer saved to {output_dir}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name to download config for")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save config to")
    args = parser.parse_args()

    download_config(args.model_name, os.path.join(args.output_dir, args.model_name))