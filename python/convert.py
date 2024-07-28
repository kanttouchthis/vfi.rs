import sys
import torch

from safetensors import torch as stt

assert len(sys.argv) == 3, f"Usage: {sys.argv[0]} <input.pkl> <output.safetensors>"

inputpkl, outputst = sys.argv[1:]
model = torch.load(inputpkl)["model"]
model = {k:v for k, v in model.items() if "_tea" not in k}

stt.save_file(model, outputst)
