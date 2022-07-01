import argparse
import opcode
import re, os
import torch

def layer_name_mapping(key):
    """Convert transformers PP in Megatron-DeepSpeed TP/PP weights mapping"""
    # Handle first and last layers
    layer_rename_map = {
        "decoder.embed_tokens.weight": "word_embeddings.weight",
        "decoder.embed_positions.weight": "embed_positions.weight",
        "decoder.final_layer_norm.weight": "weight",
        "decoder.final_layer_norm.bias": "bias",
    }

    if key in layer_rename_map:
        return "layer_01-model_00-model_states.pt" if not ("final" in key) else "layer_99-model_00-model_states.pt", layer_rename_map[key]
    # odict_keys(['input_layernorm.weight', 'input_layernorm.bias', 'self_attention.query_key_value.weight', 'self_attention.query_key_value.bias', 'self_attention.dense.weight', 'self_attention.dense.bias', 'post_attention_layernorm.weight', 'post_attention_layernorm.bias', 'mlp.dense_h_to_4h.weight', 'mlp.dense_h_to_4h.bias', 'mlp.dense_4h_to_h.weight', 'mlp.dense_4h_to_h.bias'])
    # Handle transformer blocks 
    layer_number = int(re.match(r".*decoder.layers.(\d*).*", key)[1])
    meg_key = key.replace("decoder.layers."+str(layer_number)+".", "")
    layer_number += 3
    file_name = "layer_" + str(layer_number).zfill(2)+"-model_00-model_states.pt"
    return file_name, meg_key

def post_process_transformers_keys(key):
    layer_rename_map = {
        "fc1.weight": "mlp.dense_h_to_4h.weight", # but will need to change the config, actually it seems to be h to 3h 
        "fc1.bias": "mlp.dense_h_to_4h.bias",
        "fc2.weight": "mlp.dense_4h_to_h.weight",
        "fc2.bias": "mlp.dense_4h_to_h.bias",
        "self_attn_layer_norm.weight": "input_layernorm.weight", # this corresponds to this line: https://github.com/huggingface/transformers/blob/49cd736a288a315d741e5c337790effa4c9fa689/src/transformers/models/opt/modeling_opt.py#L315
        "self_attn_layer_norm.bias": "input_layernorm.bias",
        "self_attn.out_proj.weight":"self_attention.dense.weight",
        "self_attn.out_proj.bias":"self_attention.qense.bias",
        "final_layer_norm.weight":"post_inter_attention_layernorm.weight", # OPT-175 do layer norm before mlp block: https://github.com/huggingface/transformers/blob/49cd736a288a315d741e5c337790effa4c9fa689/src/transformers/models/opt/modeling_opt.py#L339
        "final_layer_norm.bias":"post_inter_attention_layernorm.bias", # there is no post-attention layer norm
    }
    if key in layer_rename_map:
        return layer_rename_map[key]
    else:
        return key

def convert_self_att(state_dict):
    state_dict["self_attention.query_key_value.weight"] = torch.cat([state_dict["self_attn.q_proj.weight"], state_dict["self_attn.k_proj.weight"], state_dict["self_attn.v_proj.weight"]], dim=1)
    state_dict["self_attention.query_key_value.bias"] = torch.cat([state_dict["self_attn.q_proj.bias"], state_dict["self_attn.k_proj.bias"], state_dict["self_attn.v_proj.bias"]], dim=0)
    
    state_dict.pop("self_attn.q_proj.weight")
    state_dict.pop("self_attn.q_proj.bias")
    state_dict.pop("self_attn.k_proj.weight")
    state_dict.pop("self_attn.k_proj.bias")
    state_dict.pop("self_attn.v_proj.weight")
    state_dict.pop("self_attn.v_proj.bias")

    return state_dict

def convert_opt_checkpoint_to_megatron(
    opt_checkpoint_path, megatron_dump_folder_path
):
    file_names = os.listdir(opt_checkpoint_path)
    file_names = list(sorted(filter(lambda s: s.startswith("pytorch_model") and ".bin" in s, file_names)))

    # We need one file per layer
    for i, file in enumerate(file_names):
        print("Processing file:", file)
        temp = torch.load(os.path.join(opt_checkpoint_path, file), map_location="cpu")
        keys = list(temp.keys())
        new_file_names = {}
        for key in keys:
            file_name, meg_key = layer_name_mapping(key)
            if file_name not in new_file_names.keys():
                new_file_names[file_name] = [(key, meg_key)]
            else:
                new_file_names[file_name].append((key, meg_key))
        for file_name in new_file_names.keys():
            print("Writing file:", file_name)
            converted_meg_tensor = {}
            is_attn = False
            for key_tuple in new_file_names[file_name]:
                key, meg_key = key_tuple
                meg_key = post_process_transformers_keys(meg_key)
                # TODO convert the qkv layers to megatron format
                converted_meg_tensor[meg_key] = temp[key]
                if "self_attn" in key:
                    is_attn = True
            if is_attn:
                converted_meg_tensor = convert_self_att(converted_meg_tensor)
                is_attn = False
            torch.save(converted_meg_tensor, os.path.join(megatron_dump_folder_path, file_name+'.pt'))


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