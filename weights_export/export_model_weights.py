import torch
import numpy as np
from gguf import GGUFWriter
from typing import Dict, Any
from dp.model.model import load_checkpoint

def parse_hparams(config: Dict[str, Any], gguf_writer: GGUFWriter):
    gguf_writer.add_int32("encoder_vocab_size", config['model']['encoder_vocab_size'])
    gguf_writer.add_int32("decoder_vocab_size", config['model']['decoder_vocab_size'])
    gguf_writer.add_int32("d_model", config['model']['d_model'])
    gguf_writer.add_int32("d_fft", config['model']['d_fft'])
    gguf_writer.add_int32("layers", config['model']['layers'])
    gguf_writer.add_float("dropout", config['model']['dropout'])
    gguf_writer.add_int32("heads", config['model']['heads'])

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

    gguf_writer = GGUFWriter(output_gguf_path, "model_name")

    # Insert hyperparameters
    parse_hparams(config, gguf_writer)

    # Insert weights
    parse_model(model, gguf_writer)

    # Save model and hyperparameters to file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print(f"Model saved to {output_gguf_path}")
