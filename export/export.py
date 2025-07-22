import torch
import numpy as np
from gguf import GGUFWriter
from typing import Dict, Any
from dp.model.model import Model, load_checkpoint
from dp.preprocessing.text import Preprocessor

def parse_hparams(config: Dict[str, Any], gguf_writer: GGUFWriter, model: Model):
    # List of hyperparameters to add
    hyperparams = [
        ("languages", str),
        ("encoder_vocab_size", int),
        ("encoder_symbols", str),
        ("decoder_vocab_size", int),
        ("decoder_symbols", str),
        ("char_repeats", int),
        ("lowercase", bool),
        ("d_model", int),
        ("layers", int),
        ("heads", int),
    ]

    preprocessing = config['preprocessing']

    languages = ''
    for language in preprocessing['languages']:
        languages += language + ' '

    text_symbols = ''
    for symbol in preprocessing['text_symbols']:
        text_symbols += symbol + ' '
    print(f"Text symbols: {text_symbols}")

    phoneme_symbols = ''
    for symbol in preprocessing['phoneme_symbols']:
        phoneme_symbols += symbol + ' '
    phoneme_symbols += '. , : ; ? ! \" ( ) -'
    print(f"Phoneme symbols: {phoneme_symbols}")

    # Add each hyperparameter if it exists in the config
    for param, dtype in hyperparams:
        try:
            if param == "encoder_vocab_size":
                value = model.embedding.num_embeddings
            elif param == "decoder_vocab_size":
                value = model.fc_out.out_features
            elif param == "encoder_symbols":
                value = text_symbols.strip()
            elif param == "decoder_symbols":
                value = phoneme_symbols.strip()
            elif param == "languages":
                value = languages.strip()
            elif param == "char_repeats":
                value = preprocessing['char_repeats']
            elif param == "lowercase":
                value = preprocessing['lowercase']
            else:
                value = config['model'][param]

            if dtype == int:
                gguf_writer.add_int32(param, value)
            elif dtype == float:
                gguf_writer.add_float32(param, value)
            elif dtype == str:
                gguf_writer.add_string(param, value)
            elif dtype == bool:
                gguf_writer.add_bool(param, value)
            print(f"Added hyperparameter {param} with value {value}")
        except KeyError:
            print(f"Warning: {param} not found in config")

def parse_model(model: torch.nn.Module, gguf_writer: GGUFWriter):
    checkpoint = model.state_dict()

    for name, var_data in checkpoint.items():
        if "self_attn.in_proj_weight" in name or "self_attn.in_proj_bias" in name:
            # Split QKV weights/biases into separate tensors
            layer_prefix = name.rsplit(".self_attn.in_proj_", 1)[0]
            is_bias = "bias" in name
            var_data = var_data.cpu().numpy().astype(np.float32)

            d_model = var_data.shape[1] if not is_bias else var_data.shape[0] // 3

            if is_bias:
                bq = var_data[0 * d_model:1 * d_model]
                bk = var_data[1 * d_model:2 * d_model]
                bv = var_data[2 * d_model:3 * d_model]
                gguf_writer.add_tensor(f"{layer_prefix}.self_attn.q_proj.bias", bq)
                gguf_writer.add_tensor(f"{layer_prefix}.self_attn.k_proj.bias", bk)
                gguf_writer.add_tensor(f"{layer_prefix}.self_attn.v_proj.bias", bv)
                print(f"[split bias] {layer_prefix}.self_attn.[q/k/v]_proj.bias → shape: ({d_model},)")
            else:
                wq = var_data[0 * d_model:1 * d_model, :]
                wk = var_data[1 * d_model:2 * d_model, :]
                wv = var_data[2 * d_model:3 * d_model, :]
                gguf_writer.add_tensor(f"{layer_prefix}.self_attn.q_proj.weight", wq)
                gguf_writer.add_tensor(f"{layer_prefix}.self_attn.k_proj.weight", wk)
                gguf_writer.add_tensor(f"{layer_prefix}.self_attn.v_proj.weight", wv)
                print(f"[split weight] {layer_prefix}.self_attn.[q/k/v]_proj.weight → shape: ({d_model}, {var_data.shape[1]})")
        else:
            var_data = var_data.cpu().numpy().astype(np.float32)
            gguf_writer.add_tensor(name, var_data)
            print(f"[tensor] {name} shape: {var_data.shape}")

    # Export pos_encoding manually
    if hasattr(model, 'pos_encoder') and hasattr(model.pos_encoder, 'pe'):
        pe = model.pos_encoder.pe  # [5000, 1, 512]
        pe = pe.squeeze(1)         # [5000, 512]
        pe = pe[:64]               # [64, 512]
        pe = pe.reshape(1, 1, 64, 512)  # [1, 1, 64, 512]
        gguf_writer.add_tensor("pos_encoding", pe.numpy().astype(np.float32))

if __name__ == "__main__":
    checkpoint_path = "./en_us_cmudict_ipa_forward.pt"
    output_gguf_path = "./deep_phonemizer.gguf"
    device = "cpu"

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint['config']

    print(f"Config: {config}")

    gguf_writer = GGUFWriter(output_gguf_path, "model_name")

    # Insert hyperparameters
    parse_hparams(config, gguf_writer, model)

    # Insert weights
    parse_model(model, gguf_writer)

    for name in model.state_dict().keys():
        print(name)

    # Save model and hyperparameters to file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print(f"Model saved to {output_gguf_path}")
