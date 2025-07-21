import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"

import tvm
from tvm.relax.frontend.torch import from_exported_program

import torch
import tvm.meta_schedule as ms
from torch import nn
from torch.export import export

import numpy as np

from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I

######################################################################
# Define the IRModule
M, K, N = 32, 64, 128
dtype = "float32"


torchdtype2tvmdtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.matmul1 = torch.nn.Linear(
            in_features=K, out_features=N, bias=False, dtype=torchdtype2tvmdtype[dtype]
        )
        self.matmul2 = torch.nn.Linear(
            in_features=N, out_features=N, bias=False, dtype=torchdtype2tvmdtype[dtype]
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        lv0 = self.matmul1(x)
        lv1 = self.matmul2(lv0)
        lv2 = self.relu(lv1)
        return lv2


torch_data = torch.randn(M, K, dtype=torchdtype2tvmdtype[dtype])
example_args = (torch_data,)
with torch.no_grad():
    exported_program = torch.export.export(CNNModule().eval(), example_args)
    mod_from_torch = from_exported_program(exported_program, keep_params_as_input=True)


def flatten_params(tvm_params, param_names):
    params = {}
    for k, v in tvm_params.items():
        count = 0
        if isinstance(v, list):
            for array in v:
                params[param_names[count]] = array
                count += 1
        elif isinstance(v, tvm.nd.NDArray):
            params[param_names[count]] = v
            count += 1
        else:
            raise NotImplementedError("Unsupported type: {}".format(type(v)))
    return params


# tvm_params is p_matmul_weight in the exported TVM IRModule.
mod, tvm_params = tvm.relax.frontend.detach_params(mod_from_torch)
# The first param is x of forward.
tvm_params = flatten_params(tvm_params, mod["main"].params[1:])


######################################################################
# Define the target
target = tvm.target.Target("nvidia/nvidia-h100")
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tuning_logs")


######################################################################
# Optimize relax module
pipeline = tvm.relax.get_pipeline(
    name="static_shape_tuning",
    total_trials=4,
    target=target,
    work_dir=work_dir,
    cpu_weight_prepack=True,
    max_trials_per_task=1,
)
mod = pipeline(mod)
mod.show()


######################################################################
# Build
dev = tvm.cuda()
exec_relax = tvm.compile(mod, target)
vm = tvm.relax.VirtualMachine(exec_relax, dev)


######################################################################
# Prepare Data
x_np = np.random.rand(M, K).astype(dtype)
matmul1_weight_np = np.random.rand(K, N).astype(dtype)
matmul2_weight_np = np.random.rand(N, N).astype(dtype)
x_tvm = tvm.nd.array(x_np, device=dev)
matmul1_weight_tvm = tvm.nd.array(matmul1_weight_np, device=dev)
matmul2_weight_tvm = tvm.nd.array(matmul2_weight_np, device=dev)
y_tvm = tvm.nd.array(np.zeros((M, N), dtype=dtype), device=dev)


######################################################################
# Execute
y_tvm = vm["main"](x_tvm, matmul1_weight_tvm, matmul2_weight_tvm)
assert y_tvm[0].numpy().shape == (M, N)
y_tvm = np.array([[y_tvm[0].numpy()[i][j] for j in range(N)] for i in range(M)])


######################################################################
# Testing
model = CNNModule().cuda()
model.matmul1.weight.data = torch.from_numpy(matmul1_weight_np).cuda().transpose(0, 1)
model.matmul2.weight.data = torch.from_numpy(matmul2_weight_np).cuda().transpose(0, 1)
x = torch.from_numpy(x_np).cuda()
np_result = model(x).detach().cpu().numpy()
np.testing.assert_allclose(y_tvm, np_result, rtol=1e-1)
print("Pass!")


######################################################################
# Output
print("TVM output:")
print(y_tvm[:5, :5])
print("NumPy output:")
print(np_result[:5, :5])
