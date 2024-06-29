import torch
import torch.nn.functional as F
from model_dinov2 import Dinov2ForImageClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./checkpoint_dinov2/best_model.pt"

# # Create quantize
# model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

# model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


# # khởi tạo quantize model instance
# model_int8 = torch.quantization.quantize_dynamic(
#     model,  # model gốc
#     {torch.nn.Linear,
#      torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
#     dtype=torch.qint8)  # target dtype cho các quantized weights

# # run model
# input_fp32 = torch.randn(1, 3, 32, 32)
# res = model_int8(input_fp32)
# print(res)
# print(model_int8.parameters)

# torch.save(model_int8.state_dict(), './checkpoint_dinov2/best_model_quantize.pt')

# Load quantize
model_qt = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)
model_qt.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model_qt = torch.quantization.quantize_dynamic(
    model_qt,  # model gốc
    {torch.nn.Linear,
     torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
    dtype=torch.qint8)  # target dtype cho các quantized weights
model_qt.load_state_dict(torch.load('./checkpoint_dinov2/best_model_quantize.pt', map_location=torch.device(device)))
# print(model_qt.parameters)
input_fp32 = torch.randn(1, 3, 224, 224)
res = model_qt(input_fp32)
print(res)
