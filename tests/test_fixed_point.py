import sys
sys.path.append('./python')

import pytest


import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd

from typing import Tuple

import numpy as np

np.random.seed(1)


def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]


_DEVICES = [
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]



class TestOp(ndl.TensorOp):
    def compute(self, *args: Tuple[nd.NDArray]):
        z, x, W = args
        return ndl.array_api.tanh(z @ W + x)

    def gradient(self, out_grad: ndl.Tensor, node: ndl.Tensor):
        z, x, W = node.inputs
        x1 = z @ W
        x2 = x1 + x
        x3 = ndl.tanh(x2)

        out_grad3 = x3.op.gradient_as_tuple(out_grad, x3)[0]
        out_grad2_x1, out_grad2_x = x2.op.gradient_as_tuple(out_grad3, x2)
        out_grad_z, out_grad_W = x1.op.gradient_as_tuple(out_grad2_x1, x1)

        return out_grad_z, out_grad2_x, out_grad_W


class TestModule(nn.Module):
    def __init__(self, out_features: int, device):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False, device=device)
        self.tanh = nn.Tanh()

    def forward(self, Z: ndl.Tensor, X: ndl.Tensor) -> ndl.Tensor:
        return self.tanh(self.linear(Z) + X)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_solver_sanity_check(device):
    x = ndl.array_api.full((2, 2), 0, device=device, dtype="float32")
    w = ndl.array_api.array(np.random.randn(2, 2), device=device, dtype="float32")
    solver = ndl.solver.ForwardIteration()
    Z = solver.solve(lambda Z: ndl.array_api.tanh(Z @ w + x), x)
    assert Z.sum() == 0


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fixed_point(device):
    np.random.seed(1)
    _x = ndl.init.randn(2, 2, dtype="float32")
    _W = ndl.init.kaiming_uniform(fan_in=2, fan_out=2, dtype="float32")

    x = ndl.Tensor(_x, device=device)
    W = ndl.Tensor(_W, device=device)

    op = TestOp()
    solver = ndl.solver.ForwardIteration(debug=True)
    res = ndl.fixed_point(x, W, op=op, solver=solver)

    assert solver.meta.norms[-1] < 1e-4
    assert np.linalg.norm(res.numpy() - op(res, x, W).numpy()) < 1e-4

    # just checking that we didn't break auto diff
    res.sum().backward()

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fixed_point_module(device):
    np.random.seed(1)
    _x = ndl.init.randn(2, 2, dtype="float32")

    x = ndl.Tensor(_x, device=device)

    op = ndl.ModuleOp(TestModule(2, device=device))
    solver = ndl.solver.ForwardIteration(debug=True)
    res = ndl.fixed_point(x, *op.module.parameters(), op=op, solver=solver)

    assert solver.meta.norms[-1] < 1e-4
    assert np.linalg.norm(res.numpy() - op(res, x).numpy()) < 1e-4

    # just checking that we didn't break auto diff
    res.sum().backward()

    assert [p.grad is not None for p in op.module.parameters()]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fixed_point_backward(device):
    np.random.seed(1)
    _x = ndl.init.randn(2, 2, dtype="float32")
    _W = ndl.init.kaiming_uniform(fan_in=2, fan_out=2, dtype="float32")

    x = ndl.Tensor(_x , device=device)
    W = ndl.Tensor(_W, device=device)

    backward_check(ndl.fixed_point, x, W, op=TestOp(), solver=ndl.solver.ForwardIteration())


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fixed_point_backward_module(device):
    np.random.seed(1)
    _x = ndl.init.randn(2, 2, dtype="float32")

    x = ndl.Tensor(_x , device=device)

    op = ndl.ModuleOp(TestModule(2, device=device))

    backward_check(ndl.fixed_point, x, *op.module.parameters(), op=op, solver=ndl.solver.ForwardIteration())


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_test_op_backward(device):
    _z = np.random.randn(2, 2).astype(np.float32)
    _x = np.random.randn(2, 2).astype(np.float32)
    _W = np.random.randn(2, 2).astype(np.float32)

    z = ndl.Tensor(nd.array(_z), device=device)
    x = ndl.Tensor(nd.array(_x), device=device)
    W = ndl.Tensor(nd.array(_W), device=device)

    backward_check(TestOp(), z, x, W)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_op_from_module_backward(device):
    _z = np.random.randn(2, 2).astype(np.float32)
    _x = np.random.randn(2, 2).astype(np.float32)

    z = ndl.Tensor(nd.array(_z), device=device)
    x = ndl.Tensor(nd.array(_x), device=device)

    op = ndl.ModuleOp(TestModule(2, device=device))
    backward_check(op, z, x, *op.module.parameters())
