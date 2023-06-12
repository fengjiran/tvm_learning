import torch
import torchvision
import onnx
from PIL import Image
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

# from tvm._ffi.registry import list_global_func_names
from tvm._ffi.registry import get_global_func

# func_names = list_global_func_names()
func_name =  'relay.transform.LiftConstants'
func = get_global_func(func_name)
onnx_model = onnx.load('resnet18.onnx')

image_path = 'cat.png'
img = Image.open(image_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

# 加载torchvision中的ResNet18模型
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
x = np.expand_dims(img, 0)

# 使用Pytorch进行预测结果
with torch.no_grad():
    torch_img = torch.from_numpy(x)
    output = model(torch_img)

    top1_torch = np.argmax(output.numpy())

print(top1_torch)

target = "llvm"

input_name = "input.1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print(mod)

with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()

print(np.argmax(tvm_output))
