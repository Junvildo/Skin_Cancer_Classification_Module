from PIL import Image
import torchvision.transforms as transforms
from torch import load
import torch
from model_dinov2 import Dinov2ForImageClassification
import warnings
import onnxruntime
import time

warnings.filterwarnings("ignore")

ort_session = onnxruntime.InferenceSession("./checkpoint_onnx/skin_cancer_classification_quant.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_path = "data/data/test/benign/melanoma_6.jpg"
img = Image.open(img_path)

resize = transforms.Resize([224, 224])
img_y = resize(img)

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
end = time.time()
print(f"Inference of ONNX model used {end - start} seconds")
img_out_y = ort_outs[0]

print(img_out_y[0])