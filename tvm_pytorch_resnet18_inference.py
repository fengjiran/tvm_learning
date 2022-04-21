import numpy as np
from PIL import Image
import time
import torch
import torchvision
from torchvision import transforms
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata

# Load a pretrained pytorch model
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

image_path = 'cat.png'
img = Image.open(image_path).resize((224, 224))

# Preprocess the image and convert to tensor
my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

# Import the graph to relay
# The input name can be arbitrary
input_name = 'input0'
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Relay build
target = tvm.target.Target('llvm', 'llvm')
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

tvm_time_spent = []
torch_time_spent = []
n_warmup = 5
n_time = 10
for i in range(n_warmup + n_time):
    dtype = 'float32'
    m = graph_executor.GraphModule(lib['default'](ctx))
    # set inputs
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # execute
    tvm_t0 = time.perf_counter()
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    tvm_time_spent.append(time.perf_counter() - tvm_t0)

synset_path = "imagenet_synsets.txt"
# "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_synsets.txt"

with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

# class_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_classes.txt",
#     ]
# )
class_path = "imagenet_classes.txt"
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.asnumpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
for i in range(n_warmup + n_time):
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        torch_t0 = time.perf_counter()
        output = model(torch_img)
        torch_time_spent.append(time.perf_counter() - torch_t0)
        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]

tvm_time = np.mean(tvm_time_spent[n_warmup:]) * 1000
torch_time = np.mean(torch_time_spent[n_warmup:]) * 1000

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
print('Relay time(ms): {:.3f}'.format(tvm_time))
print('Torch time(ms): {:.3f}'.format(torch_time))

# with torch.no_grad():
#     torch_img = torch.from_numpy(img)
#     output = model(torch_img)
#
#     top1_torch = np.argmax(output.numpy())
#
# print(top1_torch)
#
# # export onnx
#
# torch_out = torch.onnx.export(model, torch_img, 'resnet18.onnx', verbose=True, export_params=True)
