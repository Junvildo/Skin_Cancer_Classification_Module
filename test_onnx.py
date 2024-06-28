from PIL import Image
import torchvision.transforms as transforms
from torch import load
import torch
from model_dinov2 import Dinov2ForImageClassification
import warnings
import onnxruntime

warnings.filterwarnings("ignore")

ort_session = onnxruntime.InferenceSession("skin_cancer_classification.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_path = "img_path"
img = Image.open(img_path)

resize = transforms.Resize([224, 224])
img_y = resize(img)

# img_ycbcr = img.convert('YCbCr')
# img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

print(img_out_y[0])