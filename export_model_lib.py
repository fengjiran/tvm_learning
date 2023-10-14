import os
import numpy as np
from PIL import Image
from torchvision import transforms
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import relay_model


class InferenceResnet(object):
    def __init__(self,
                 batch_size=1,
                 image_shape=(3, 224, 224),
                 print_model=True,
                 opt_level=2,
                 target=tvm.target.Target("llvm"),
                 device=tvm.cpu(0),
                 lib_path="resnet_lib.tar"
                 ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.print_model = print_model
        self.opt_level = opt_level
        self.target = target
        self.device = device
        self.lib_path = lib_path

    def build(self):
        if not os.path.exists(self.lib_path):
            mod, params = relay_model.resnet.get_workload(num_layers=18,
                                                          batch_size=self.batch_size,
                                                          image_shape=self.image_shape)
            if self.print_model:
                print(mod.astext(show_meta_data=False))

            with tvm.transform.PassContext(opt_level=self.opt_level):
                lib = relay.build(mod, self.target, params=params)
            lib.export_library(self.lib_path)

    def run(self):
        if not os.path.exists(self.lib_path):
            raise ValueError("Build model first.")

        height, width = self.image_shape[1:]
        img = Image.open('cat.png').resize((height, width))
        my_preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img = my_preprocess(img)
        img = np.expand_dims(img, 0)
        input_data = tvm.nd.array(img.astype('float32'))

        lib = tvm.runtime.load_module(self.lib_path)
        module = graph_executor.GraphModule(lib['default'](self.device))
        module.set_input('data', input_data)
        module.run()

        return module.get_output(0).numpy()


if __name__ == "__main__":
    resnet = InferenceResnet()
    resnet.build()
    out = resnet.run()
    print(np.argmax(out))
    print('done')
