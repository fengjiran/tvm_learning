import numpy as np
import tvm
from tvm import relay

bs = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (bs,) + image_shape
output_shape = (bs, num_class)

mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=bs, image_shape=image_shape)
print(mod.astext(show_meta_data=False))
