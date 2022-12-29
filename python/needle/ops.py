"""Operatpr table."""
# Global operator table.
from functools import lru_cache
from itertools import zip_longest
from numbers import Number
from typing import Iterable, Optional, List, Tuple, Union

from .solver import BaseSolver
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray, *_):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return self.scalar * (node.inputs[0] ** (self.scalar - 1)) * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        if self.axes is None:
            axis1, axis2 = range(a.ndim)[-2:]
        else:
            axis1, axis2 = self.axes

        new_axes = list(range(a.ndim))
        new_axes[axis1], new_axes[axis2] = new_axes[axis2], new_axes[axis1]

        return array_api.permute(a, tuple(new_axes))

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.transpose(axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.reshape(shape=node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    @staticmethod
    def _is_broadcast(input_dim: Optional[int], required_dim: int):
        """
        Define if input_dim would be broadcast by comparing with required_dim
        according to https://numpy.org/doc/stable/user/basics.broadcasting.html
        """
        return (input_dim == 1 and input_dim != required_dim) or input_dim is None


    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        if input_shape == self.shape:
            # no broadcasting <-> identity op
            return out_grad

        broadcast_axes = []
        for i, (input_dim, required_dim) in enumerate(
            reversed(list(zip_longest(input_shape[::-1], self.shape[::-1])))
        ):
            if self._is_broadcast(input_dim, required_dim):
                broadcast_axes.append(i)

        return out_grad.sum(axes=tuple(broadcast_axes)).reshape(input_shape)



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[Tuple, int]] = None):
        self.axes = axes
        if isinstance(self.axes, int):
            self.axes = (self.axes,)

    def compute(self, a: NDArray):
        if self.axes is None:
            return array_api.summation(a)

        for ax in reversed(self.axes):
            a = array_api.summation(a, ax)

        return a

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        axes = range(len(input_shape)) if self.axes is None else list(self.axes)

        # like when calling np.sum(..., keepdims=True)
        keepdims_shape = numpy.array(input_shape)
        keepdims_shape[axes] = 1

        return out_grad.reshape(tuple(keepdims_shape)).broadcast_to(input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray, *_):
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        left_grad = out_grad @ rhs.transpose()
        right_grad = lhs.transpose() @ out_grad

        # in case N-D, N > 2 we need to sum along broadcast axes
        # to ensure that shape of grad equals to shape of input
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

        left_broadcast_axes = tuple(range(len(left_grad.shape) - len(lhs.shape)))
        right_broadcast_axes = tuple(range(len(right_grad.shape) - len(rhs.shape)))

        if left_broadcast_axes:
            left_grad = left_grad.sum(left_broadcast_axes)

        if right_broadcast_axes:
            right_grad = right_grad.sum(right_broadcast_axes)

        return left_grad, right_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        flag = array_api.array(
            node.realize_cached_data() > 0,
            dtype=node.dtype,
            device=node.device,
        )
        return out_grad * Tensor.make_const(flag)

def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[Union[Tuple, int]] = None):
        self.axes = axes
        if isinstance(self.axes, int):
            self.axes = (self.axes,)

    def _get_keepdims_shape(self, input_shape: tuple):
        axes = range(len(input_shape)) if self.axes is None else list(self.axes)

        # like when calling np.sum(..., keepdims=True)
        keepdims_shape = numpy.array(input_shape)
        keepdims_shape[axes] = 1

        return keepdims_shape

    def _get_keepdims_array(self, res: NDArray, input_: NDArray):
        return array_api.reshape(res.compact(), self._get_keepdims_shape(input_.shape))

    def _get_keepdims_tensor(self, res: Tensor, input_: Tensor):
        return res.reshape(self._get_keepdims_shape(input_.shape))

    def compute(self, Z: NDArray):
        # return array_api.log(array_api.sum(array_api.exp(Z - array_api.reduce_max(Z, self.axes, keepdims=True)), self.axes)) + array_api.max(Z, self.axes)
        max_Z = array_api.reduce_max(Z, self.axes)
        broadcast_max_Z = array_api.broadcast_to(self._get_keepdims_array(max_Z, Z), Z.shape)

        return array_api.log(array_api.summation(array_api.exp(Z - broadcast_max_Z), self.axes)) + max_Z

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_node = node.inputs[0]
        return (
            self._get_keepdims_tensor(out_grad, input_node).broadcast_to(input_node.shape) *
             exp(input_node - self._get_keepdims_tensor(node, input_node).broadcast_to(input_node.shape))
        )


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray):
        return array_api.tanh(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * (1 - node ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: Iterable[NDArray]) -> NDArray:
        shape = args[0].shape
        size = args[0].size

        stacked_array = array_api.empty(
            (len(args), size),
            dtype=args[0].dtype,
            device=args[0].device,
        )
        
        for i, arg in enumerate(args):
            stacked_array[i, :] = arg.compact().reshape((1, size))

        new_axes = list(range(1, len(shape) + 1))
        new_axes.insert(self.axis, 0)

        return stacked_array.reshape((len(args), *shape)).permute(new_axes).compact()

    def gradient(self, out_grad: Tensor, node: Tensor):
        return split(out_grad, axis=self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: NDArray):
        num_of_args = A.shape[self.axis]
        size = A.size // num_of_args

        new_axes = list(range(A.ndim))
        new_axes.pop(self.axis)
        new_axes.insert(0, self.axis)

        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        
        unstacked_A = A.permute(new_axes).compact().reshape((num_of_args, size))

        arrays = tuple(
            array_api.array(unstacked_A[i, :]).reshape(new_shape) 
            for i in range(num_of_args)
        )

        return arrays


    def gradient(self, out_grad: Tensor, node: TensorTuple):
        return stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(a.shape)))

        return array_api.flip(a, axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        new_shape = []
        slices = []
        for ax in range(a.ndim):
            flag = ax in self.axes
            new_shape.append(a.shape[ax] * (1 + self.dilation * flag))
            slices.append(slice(None, None, 1 + self.dilation * flag))

        out = array_api.full(tuple(new_shape), fill_value=0, dtype=a.dtype, device=a.device)
        out[tuple(slices)] = a
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        slices = []
        for ax in range(a.ndim):
            flag = ax in self.axes
            slices.append(slice(None, None, 1 + self.dilation * flag))

        return a[tuple(slices)]

    def gradient(self, out_grad: Tensor, node: Tensor):
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A: NDArray, B: NDArray):
        pad_A = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
        )

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = pad_A.strides
        out_H = (H + 2 * self.padding - K) // self.stride + 1
        out_W = (W + 2 * self.padding - K) // self.stride + 1
        
        im2col = pad_A.as_strided(
            shape=(N, out_H, out_W, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
        ).compact().reshape((N * out_H * out_W, K * K * C_in))

        out = im2col @ B.compact().reshape((K * K * C_in, C_out))

        return out.reshape((N, out_H, out_W, C_out))

    def gradient(self, out_grad: Tensor, node: Tensor):
        # X, X_grad: N, H, W, C_in
        # w, w_grad: K, K, C_in, C_out
        # conv(X, w, s, p), out_grad: N, (H + 2p - K) // s + 1, (W + 2p - K) // s + 1, C_out
        
        X, w = node.inputs

        N, H, W, C_in = X.shape
        K, _, _, C_out = w.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        X_grad = conv(
            out_grad, 
            # transose: (C_in, C_out) --> (C_out, C_in)
            flip(w, axes=(0, 1)).transpose((2 ,3)),  
            # (H + 2p - K + 1) + 2(new_p) - K + 1 = H --> new_P = K - 1 - p
            padding=max(K - 1 - self.padding, 0),  
        )

        w_grad = conv(
            # N H W C_in --> C_in H W N
            X.transpose((0, 3)),
            # N ou_H out_W C_out --> out_H ot_W N C_out
            out_grad.transpose((0, 2)).transpose((0, 1)),
            # H + 2(new_p) - (H + 2p - K + 1) + 1 = K_H = K
            padding=self.padding,
        # C_in new_H new_W C_out --> new_H new_W C_in C_out == 
        # == K_H K_W C_in C_out == K K C_in C_out
        ).transpose((0, 2)).transpose((0, 1))

        return X_grad, w_grad
  


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)


##### fixed point #####

from .nn import Module


class FixedPoint(TensorOp):
    def __init__(self, op: Union[TensorOp, Module], solver: BaseSolver): 
        self.op = self._init_op(op)
        self.solver = solver
        self.history = []
    
    @staticmethod
    def _init_op(op: Union[TensorOp, Module]) -> TensorOp:
        if isinstance(op, TensorOp):
            return op
        elif isinstance(op, Module):
            return ModuleOp(op)
        else:
            raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        inp, *params = args
        init_value = array_api.full(
            inp.shape,
            fill_value=0,
            device=inp.device,
            dtype=inp.dtype,
        )
        res = self.solver.solve(lambda z: self.op.compute(z, inp, *params), init_value)
        self.history.append(self.solver.meta)
        return res

    def gradient(self, out_grad: Tensor, node: Tensor):
        inp, *params = node.inputs
        device = node.device
        dtype = node.dtype

        node = self.op(node, inp, *params)
        
        def _implicit_adjoint_op(v_array: NDArray) -> NDArray:
            v = Tensor.make_const(v_array)
            output_grad = self.op.gradient_as_tuple(out_grad=v, node=node)[0]
            return (out_grad + output_grad).realize_cached_data()

        implicit_output_grad_array = self.solver.solve(
            _implicit_adjoint_op,
            array_api.full(out_grad.shape, fill_value=0, device=device, dtype=dtype),
        )

        implicit_output_grad = Tensor(implicit_output_grad_array, device=device, dtype=dtype)

        return self.op.gradient_as_tuple(out_grad=implicit_output_grad, node=node)[1:]


def fixed_point(*args, op, solver):
    return FixedPoint(op, solver)(*args)


class ModuleOp(TensorOp):
    def __init__(self, module: Module):
        self.module = module

    def compute(self, *args: Tuple[NDArray]):
        z, x, *_ = args
        return self.module(Tensor.make_const(z), Tensor.make_const(x)).realize_cached_data()

    def _compute_gradient(self, out_grad: Tensor, node: Tensor, inputs: List[Tensor]):
        tmp = {node: out_grad}
        res = {}

        while tmp:
            out = next(iter(tmp))
            grad = tmp.pop(out)

            if out in inputs:
                continue

            inp_grads = out.op.gradient_as_tuple(grad, out)
            for inp, inp_grad in zip(out.inputs, inp_grads):
                tmp[inp] = inp_grad
                if inp in inputs:
                    res[inp] = inp_grad
                    
        return tuple(res[inp] for inp in inputs)

    @lru_cache(1)
    def _compute_module(self, *args: Tuple[Tensor]):
        return self.module(*args)

    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]:
        z, x, *parameters = node.inputs  # self.module.parameters
        res = self._compute_module(z, x)
        return self._compute_gradient(out_grad, res, [z, x, *parameters])


def module_op(*args, module: Module):
    return module_op(module)(*args)
