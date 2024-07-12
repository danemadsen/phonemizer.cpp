import torch
import numpy as np
from gguf import GGUFWriter
from typing import Dict, Any
from dp.model.model import load_checkpoint
from dp.preprocessing.text import Preprocessor

def parse_hparams(config: Dict[str, Any], gguf_writer: GGUFWriter, preprocessor: Preprocessor):
    # List of hyperparameters to add
    hyperparams = [
        ("encoder_vocab_size", int),
        ("decoder_vocab_size", int),
        ("d_model", int),
        ("d_fft", int),
        ("layers", int),
        ("dropout", float),
        ("heads", int),
    ]

    # Add each hyperparameter if it exists in the config
    for param, dtype in hyperparams:
        try:
            if param == "encoder_vocab_size":
                value = preprocessor.text_tokenizer.vocab_size
            elif param == "decoder_vocab_size":
                value = preprocessor.phoneme_tokenizer.vocab_size
            else:
                value = config['model'][param]

            if dtype == int:
                gguf_writer.add_int32(param, value)
            elif dtype == float:
                gguf_writer.add_float32(param, value)
            print(f"Added hyperparameter {param} with value {value}")
        except KeyError:
            print(f"Warning: {param} not found in config")

def parse_model(model: torch.nn.Module, gguf_writer: GGUFWriter):
    checkpoint = model.state_dict()
    for name, var_data in checkpoint.items():
        var_data = var_data.cpu().numpy().squeeze().astype(np.float32)
        gguf_writer.add_tensor(name, var_data)
        print(f"Processing variable: {name} with shape: {var_data.shape}")

if __name__ == "__main__":
    checkpoint_path = "./en_us_cmudict_ipa_forward.pt"
    output_gguf_path = "./deep_phonemizer.gguf"
    device = "cpu"

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint['config']

    print(f"Config: {config}")

    preprocessor = Preprocessor.from_config(config)

    gguf_writer = GGUFWriter(output_gguf_path, "model_name")

    # Insert hyperparameters
    parse_hparams(config, gguf_writer, preprocessor)

    # Insert weights
    parse_model(model, gguf_writer)

    # Save model and hyperparameters to file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print(f"Model saved to {output_gguf_path}")
