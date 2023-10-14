import numpy as np
from PIL import Image
from torchvision import transforms
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import relay_model

bs = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (bs,) + image_shape
output_shape = (bs, num_class)

img = Image.open('cat.png').resize((224, 224))
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

mod, params = relay_model.resnet.get_workload(num_layers=18, batch_size=bs, image_shape=image_shape)
print(mod.astext(show_meta_data=False))

opt_level = 3
target = tvm.target.Target('llvm')
dev = tvm.cpu(0)
# target = tvm.target.cuda()
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)
# lib.export_library("deploy_lib.tar")
module = graph_executor.GraphModule(lib['default'](dev))
module.set_input('data', tvm.nd.array(img.astype('float32')))
module.run()
out = module.get_output(0).numpy()
print(np.argmax(out))
print('done')
