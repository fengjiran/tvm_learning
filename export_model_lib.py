import numpy as np
import tvm
from tvm import relay
import relay_model

bs = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (bs,) + image_shape
output_shape = (bs, num_class)

mod, params = relay_model.resnet.get_workload(num_layers=18, batch_size=bs, image_shape=image_shape)
print(mod.astext(show_meta_data=False))

opt_level = 3
target = tvm.target.Target('llvm', 'llvm')
# target = tvm.target.cuda()
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

lib.export_library("deploy_lib.tar")

print('done')