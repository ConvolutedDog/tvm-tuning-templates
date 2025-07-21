import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"

import tvm
import numpy as np
import tvm.meta_schedule as ms

from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I

######################################################################
# Define the IRModule
M, K, N = 32, 64, 128
dtype = "float32"


@I.ir_module
class ModuleMatMul:
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [K, N], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):  # type: ignore
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # type: ignore
                with T.init():
                    C[vi, vj] = T.cast(0.0, dtype=dtype)  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


mod = ModuleMatMul


######################################################################
# Define the target
target = tvm.target.Target("nvidia/nvidia-h100")
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tuning_logs")


######################################################################
# Optimize matmul
database_matmul = ms.tir_integration.tune_tir(
    mod=mod,
    target=target,
    work_dir=work_dir,
    max_trials_global=4,
    num_trials_per_iter=4,
    # num_tuning_cores=1,
)

if database_matmul is None:
    raise ValueError("Database is None")

sch_matmul = ms.tir_integration.compile_tir(
    database=database_matmul, mod=mod["matmul"], target=target
)
if sch_matmul is None:
    print("No valid schedule found!")
else:
    sch_matmul.mod.show()
    sch_matmul.trace.show()

    ##################################################################
    # Build
    dev = tvm.cuda()
    mod = sch_matmul.mod
    ex = tvm.compile(mod, target=target)

    ##################################################################
    # Prepare Data
    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(np.zeros((M, N), dtype=dtype), device=dev)

    ##################################################################
    # Execute
    ex(a_tvm, b_tvm, c_tvm)

    ##################################################################
    # Testing
    np_result = np.dot(a_np, b_np)
    np.testing.assert_allclose(c_tvm.numpy(), np_result, rtol=1e-3)
    print("Pass!")

    ##################################################################
    # Output
    print("TVM output:")
    print(c_tvm.numpy()[:5, :5])
    print("NumPy output:")
    print(np_result[:5, :5])
