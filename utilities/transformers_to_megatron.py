import argparse
import opcode
import re, os
import torch

def layer_name_mapping(key, file):
    """Convert transformers PP in Megatron-DeepSpeed TP/PP weights mapping"""
    # Handle first and last layers
    layer_rename_map = {
        "word_embeddings.weight": "word_embeddings.weight",
        "word_embeddings_layernorm.weight": "word_embeddings.norm.weight",
        "word_embeddings_layernorm.bias":  "word_embeddings.norm.bias",
        "ln_f.weight": "weight",
        "ln_f.bias": "bias",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]
    
    # Handle transformer blocks 
    layer_number = int(re.match(r".*h.(\d*).*", file)[1])
    layer_number += 3
    return f"layer_{layer_number}.{key}"

def convert_opt_checkpoint_to_megatron(
    opt_checkpoint_path, megatron_dump_folder_path
):
    file_names = os.listdir(opt_checkpoint_path)
    file_names = list(sorted(filter(lambda s: s.startswith("pytorch-model") and ".bin" in s, file_names)))

    for i, file in enumerate(file_names):
        print("Processing file:", file)
        temp = torch.load(os.path.join(opcode, file), map_location="cpu")
        keys = list(temp.keys())
        for key in keys:
            temp[layer_name_mapping(key, file)] = temp.pop(key)
        
        torch.save(temp, os.path.join(megatron_dump_folder_path, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--opt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the transformers OPT checkpoint path.",
    )
    parser.add_argument(
        "--megatron_dump_folder_path", default=None, type=str, required=True, help="Path to the output Megatron-DS model."
    )
    args = parser.parse_args()
    convert_opt_checkpoint_to_megatron(args.opt_checkpoint_path, args.megatron_dump_folder_path)