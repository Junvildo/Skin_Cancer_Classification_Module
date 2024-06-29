import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'checkpoint_onnx/skin_cancer_classification.onnx'
model_quant = 'checkpoint_onnx/skin_cancer_classification_quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)