from torch import load
import torch
from model_dinov2 import Dinov2ForImageClassification


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./checkpoint_dinov2/best_model.pt"
model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

model.load_state_dict(load(model_path, map_location=torch.device('cpu')))
model.to(device)

# set the model to inference mode
model.eval()

# Input to the model
x = torch.randn((1, 3, 224, 224), requires_grad=True)
# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./checkpoint_onnx/skin_cancer_classification.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})