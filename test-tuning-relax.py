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
USE_GET_PIPELINE = True


######################################################################
# Define the target
target = tvm.target.Target("nvidia/nvidia-h100")
this_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(this_dir, "tuning_logs")


######################################################################
# Optimize relax module
if not USE_GET_PIPELINE:

    @I.ir_module
    class ModuleMatMulReLU:
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

        @T.prim_func
        def relu(c: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "relu", "tir.noalias": True})
            C = T.match_buffer(c, [M, N], dtype=dtype)
            Y = T.match_buffer(y, [M, N], dtype=dtype)
            for i, j in T.grid(M, N):  # type: ignore
                with T.block("Y"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Y[vi, vj] = T.max(C[vi, vj], T.cast(0.0, dtype=dtype))  # type: ignore

        @R.function
        def main(x: R.Tensor((M, K), dtype), w: R.Tensor((K, N), dtype)) -> R.Tensor((M, N), dtype):
            R.func_attr({"num_input": 1})
            cls = ModuleMatMulReLU
            with R.dataflow():
                lv0 = R.call_tir(
                    gvar=cls.matmul, args=(x, w), out_sinfo=R.Tensor((M, N), dtype=dtype)
                )
                lv1 = R.call_tir(
                    gvar=cls.relu, args=(lv0,), out_sinfo=R.Tensor((M, N), dtype=dtype)
                )
                R.output(lv1)
            return lv1

    mod = ModuleMatMulReLU
    mod.show()

    database_relax = ms.relax_integration.tune_relax(
        mod=mod,
        params={},
        target=target,
        work_dir=work_dir,
        max_trials_global=4,
        num_trials_per_iter=1,
        # num_tuning_cores=1,
    )

    if database_relax is None:
        raise ValueError("Database is None")

    exec_relax = ms.relax_integration.compile_relax(
        database=database_relax, mod=mod, target=target, params={}, enable_warning=True
    )
    if exec_relax is None:
        print("No valid VMExecutable compiled!")
    else:
        ##################################################################
        # Build
        dev = tvm.cuda()
        vm = tvm.relax.VirtualMachine(exec_relax, dev)

        ##################################################################
        # Prepare Data
        a_np = np.random.rand(M, K).astype(dtype)
        b_np = np.random.rand(K, N).astype(dtype)
        a_tvm = tvm.nd.array(a_np, device=dev)
        b_tvm = tvm.nd.array(b_np, device=dev)
        c_tvm = tvm.nd.array(np.zeros((M, N), dtype=dtype), device=dev)

        ##################################################################
        # Execute
        c_tvm = vm["main"](a_tvm, b_tvm)

        ##################################################################
        # Testing
        np_result = np.maximum(np.dot(a_np, b_np), 0.0)
        np.testing.assert_allclose(c_tvm.numpy(), np_result, rtol=1e-3)
        print("Pass!")

        ##################################################################
        # Output
        print("TVM output:")
        print(c_tvm.numpy()[:5, :5])
        print("NumPy output:")
        print(np_result[:5, :5])

else:
    # It is best not to write Relax IRModule by hand, there may be strange bugs.
    @I.ir_module
    class ModuleMatMulReLUBuiltin:
        @R.function
        def main(
            x: R.Tensor((M, K), dtype=dtype),
            p_matmul1_weight: R.Tensor((N, K), dtype=dtype),
        ) -> R.Tuple(R.Tensor((M, N), dtype=dtype)):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((K, N), dtype=dtype) = R.permute_dims(p_matmul1_weight, axes=None)
                lv1: R.Tensor((M, N), dtype=dtype) = R.matmul(x, lv, out_dtype=dtype)
                lv2: R.Tensor((M, N), dtype=dtype) = R.nn.relu(lv1)
                gv: R.Tuple(R.Tensor((M, N), dtype=dtype)) = (lv2,)
                R.output(gv)
            return gv

    mod = ModuleMatMulReLUBuiltin
    mod.show()

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

    ##################################################################
    # Prepare Data
    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(np.zeros((M, N), dtype=dtype), device=dev)

    ##################################################################
    # Execute
    c_tvm = vm["main"](a_tvm, b_tvm)
    assert c_tvm[0].numpy().shape == (M, N)
    c_tvm = np.array([[c_tvm[0].numpy()[i][j] for j in range(N)] for i in range(M)])

    ##################################################################
    # Testing
    np_result = np.maximum(np.dot(a_np, b_np), 0.0)
    np.testing.assert_allclose(c_tvm, np_result, rtol=1e-3)
    print("Pass!")

    ##################################################################
    # Output
    print("TVM output:")
    print(c_tvm[:5, :5])
    print("NumPy output:")
    print(np_result[:5, :5])
