import torch
import torch.nn.utils.prune as prune
from model_dinov2 import Dinov2ForImageClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Prune from base model
# model_path = "./checkpoint_dinov2/best_model.pt"
# model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

# model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
# model.to(device)

# for name, module in model.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.4)

# # print(dict(model.named_buffers()).keys())

# torch.save(model.state_dict(), './checkpoint_dinov2/best_model_prune.pt')

# # Prune from quantized model
# model_path = "./checkpoint_dinov2/best_model.pt"
# model_path_quantize = "./checkpoint_dinov2/best_model_quantize.pt"
# model_qt = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

# model_qt.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
# model_qt = torch.quantization.quantize_dynamic(
#     model_qt,  # model gốc
#     {torch.nn.Linear,
#      torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
#     dtype=torch.qint8)  # target dtype cho các quantized weights
# model_qt.load_state_dict(torch.load(model_path_quantize, map_location=torch.device(device)))

# for name, module in model_qt.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.4)

# # print(dict(model.named_buffers()).keys())

# torch.save(model_qt.state_dict(), './checkpoint_dinov2/best_model_quantize_prune.pt')

# # Prune from quantized model and delete org weight
# model_path = "./checkpoint_dinov2/best_model.pt"
# model_path_quantize = "./checkpoint_dinov2/best_model_quantize.pt"
# model_qt = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

# model_qt.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
# model_qt = torch.quantization.quantize_dynamic(
#     model_qt,  # model gốc
#     {torch.nn.Linear,
#      torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
#     dtype=torch.qint8)  # target dtype cho các quantized weights
# model_qt.load_state_dict(torch.load(model_path_quantize, map_location=torch.device(device)))

# for name, module in model_qt.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.4)

# # print(dict(model.named_buffers()).keys())
# # model_qt = torch.nn.utils.prune.remove(model_qt, name='weight')
# for name, module in model_qt.named_modules():
#     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         prune.remove(module, 'weight')
# torch.save(model_qt.state_dict(), './checkpoint_dinov2/best_model_quantize_prune_remove_org.pt')

# # Prune from quantized model and delete org weight structure
# model_path = "./checkpoint_dinov2/best_model.pt"
# model_path_quantize = "./checkpoint_dinov2/best_model_quantize.pt"
# model_qt = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

# model_qt.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
# model_qt = torch.quantization.quantize_dynamic(
#     model_qt,  # model gốc
#     {torch.nn.Linear,
#      torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
#     dtype=torch.qint8)  # target dtype cho các quantized weights
# model_qt.load_state_dict(torch.load(model_path_quantize, map_location=torch.device(device)))

# for name, module in model_qt.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.ln_structured(module, name='weight', amount=0.2, dim=1, n=float('-inf'))
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.ln_structured(module, name='weight', amount=0.4, dim=1, n=float('-inf'))

# # print(dict(model.named_buffers()).keys())
# # model_qt = torch.nn.utils.prune.remove(model_qt, name='weight')
# for name, module in model_qt.named_modules():
#     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         prune.remove(module, 'weight')
# torch.save(model_qt.state_dict(), './checkpoint_dinov2/best_model_quantize_prune_remove_org_structure.pt')

# # Prune from quantized model, delete org weight and convert the pruned weights to a sparse format
# model_path = "./checkpoint_dinov2/best_model.pt"
# model_path_quantize = "./checkpoint_dinov2/best_model_quantize.pt"
# model_qt = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=2)

# model_qt.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
# model_qt = torch.quantization.quantize_dynamic(
#     model_qt,  # model gốc
#     {torch.nn.Linear,
#      torch.nn.Conv2d},  # một tập hợp các layers cho dynamically quantize
#     dtype=torch.qint8)  # target dtype cho các quantized weights
# model_qt.load_state_dict(torch.load(model_path_quantize, map_location=torch.device(device)))

# for name, module in model_qt.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.4)

# # print(dict(model.named_buffers()).keys())
# # model_qt = torch.nn.utils.prune.remove(model_qt, name='weight')
# # Remove pruning masks
# for name, module in model_qt.named_modules():
#     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         prune.remove(module, 'weight')

# # Convert the pruned weights to a sparse format
# for name, module in model_qt.named_modules():
#     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         weight = module.weight.data
#         mask = weight != 0
#         module.weight = torch.nn.Parameter(weight[mask].to_sparse())
# torch.save(model_qt.state_dict(), './checkpoint_dinov2/best_model_quantize_prune_remove_org_spare_format.pt')
