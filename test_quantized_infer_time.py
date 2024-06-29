from torch import nn, load
import torch
from model_dinov2 import Dinov2ForImageClassification
import os
import gc
import warnings
import time


warnings.filterwarnings("ignore")

# Input to the model


device = "cuda" if torch.cuda.is_available() else "cpu"
model_list = os.listdir("./checkpoint_dinov2_optim")
model_names = [name.replace('.pt', '') for name in model_list]
model_paths = [os.path.join("./checkpoint_dinov2_optim", model_path) for model_path in model_list]
for model_path, model_name in zip(model_paths, model_names):
    if 'spare' in model_name:
        continue
    print(model_name)
    print("="*50)
    model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)
    model = torch.quantization.quantize_dynamic(
        model,  # model gốc
        {torch.nn.Linear,
        torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
        dtype=torch.qint8)  # target dtype cho các quantized weights
    model.load_state_dict(load(model_path, map_location=torch.device(device)))
    model.eval()
    x = torch.randn((8, 3, 224, 224), requires_grad=True)
    start = time.time() 
    torch_out = model(x)
    end = time.time()
    print(f"Inference of Pytorch with {model_name} used {end - start} seconds")

    del model
    gc.collect()