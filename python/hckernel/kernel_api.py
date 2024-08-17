# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Iterable
from functools import partial
from greenlet import greenlet
from itertools import product
from numpy import ma
from sympy import Symbol
from sympy.core.expr import Expr
from math import prod
import inspect
import numpy as np
import types
import typing
import functools
import operator

from .indexing import sym
from .typename import typename
from .symbol_registry import register_symbol as _reg_symbol_impl


def _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables):
    assert (
        subgroup_size is None
        or isinstance(subgroup_size, int)
        or subgroup_size in literals
    ), "Subgroup size must be const or literal"


class TunableParam:
    def __init__(self, sym, default, vals, strategy=None):
        self.sym = sym
        self.default = default
        self.vals = vals
        self.strategy = strategy


def resolve_symbols(func, symbols):
    old_closure = func.__closure__
    new_closure = None

    def resolve_impl(val):
        if isinstance(val, Symbol):
            if val in symbols.keys():
                return symbols[val]
        elif isinstance(val, Expr):
            return val.subs(symbols)
        elif isinstance(val, Mapping):
            return Mapping(resolve_symbols(val.func, symbols))

        return None

    if old_closure is not None:
        cell_cls = type(old_closure[0])

        def resolve_cell(cell):
            res = resolve_impl(cell.cell_contents)
            if res is None:
                res = cell
            else:
                res = cell_cls(res)

            return res

        new_closure = tuple(resolve_cell(cell) for cell in old_closure)

    def resolve_global(val):
        res = resolve_impl(val)
        if res is None:
            res = val

        return res

    new_globals = {key: resolve_global(val) for key, val in func.__globals__.items()}

    g = types.FunctionType(
        func.__code__,
        new_globals,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=new_closure,
    )
    g = functools.update_wrapper(g, func)
    g.__kwdefaults__ = func.__kwdefaults__
    return g


def _not(val):
    if isinstance(val, (bool, int)):
        return val == 0

    if isinstance(val, np.ndarray):
        return ~val

    if isinstance(val, (tuple, list)):
        return type(val)([_not(v) for v in val])

    assert False


class _masked_array:
    def __init__(self, data, mask=True):
        if isinstance(data, ma.masked_array):
            assert mask
            self.__ma = data
        else:
            self.__ma = ma.masked_array(data, mask=_not(mask))

    def __set_mask(self, key, value):
        self.__ma.mask[key] = _not(value)

    @property
    def mask(self):
        return _not(self.__ma.mask)

    @property
    def data(self):
        return self.__ma.data

    def __setitem__(self, key, value):
        if isinstance(value, _masked_array):
            self.data[key] = np.where(value.mask, value.data, self.data[key])
            self.__set_mask(key, self.mask[key] | value.mask)
        else:
            self.data[key] = value
            self.__set_mask(key, True)

    def __getitem__(self, key):
        return _masked_array(self.__ma[key])

    def _bin_op_implementation(self, other, op):
        if isinstance(other, _masked_array):
            return _masked_array(op(self.__ma, other.__ma))
        else:
            return _masked_array(op(self.__ma, other))

    def _bin_iop_implementation(self, other, op):
        self[:] = op(self, other)

        return self

    def _un_op_implemenation(self, op):
        return _masked_array(op(self.__ma))

    def __getattr__(self, name):
        return getattr(self.__ma, name)

    def __repr__(self):
        return str(self.__ma)


def set_arithm_methods():
    bin_op_list = [
        operator.add,
        operator.sub,
        operator.mul,
        operator.matmul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
        operator.mod,
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.rshift,
        operator.lshift,
    ]

    un_op_list = [operator.abs, operator.neg, operator.not_, operator.inv]

    def _name(op):
        return op.__name__.replace("_", "")

    def set_bin_attr(op):
        setattr(
            _masked_array,
            f"__{_name(op)}__",
            lambda self, other: self._bin_op_implementation(other, op),
        )
        setattr(
            _masked_array,
            f"__i{_name(op)}__",
            lambda self, other: self._bin_iop_implementation(other, op),
        )

    def set_un_attr(op):
        setattr(
            _masked_array,
            f"__{_name(op)}__",
            lambda self: self._un_op_implemenation(op),
        )

    for op in bin_op_list:
        set_bin_attr(op)

    for op in un_op_list:
        set_un_attr(op)


set_arithm_methods()


class Mapping:
    def __init__(self, func):
        self.func = func


def create_mapping(func):
    return Mapping(func)


def _get_uninit_value(dtype):
    return 0


def _divup(a, b):
    return (a + b - 1) // b


class CurrentSubGroup:
    def __init__(self, size, subgroup_id):
        self._size = size
        self._subgroup_id = subgroup_id

    def size(self):
        return self._size

    def subgroup_id(self):
        return self._subgroup_id


class CurrentWorkitem:
    def __init__(self, lid, gid):
        self._local_id = lid
        self._global_id = gid

    def local_id(self):
        return self._local_id

    def global_id(self):
        return self._global_id


class CurrentGroup:
    def __init__(self, group_shape, subgroup_size):
        self._dims = len(group_shape)
        self._group_shape = group_shape
        self._subgroup_size = subgroup_size
        self._subgroup_ranges = self.__get_subgroup_ranges()
        self._workitem_ranges = self.__get_workitems_ranges()
        self._group_id = None
        self._tasks = []
        self._current_task = 0

    def __get_subgroup_ranges(self):
        rngs = []
        if len(self._group_shape) > 1:
            rngs += [range(v) for v in self._group_shape[0:-1]]

        rngs += [range(_divup(self._group_shape[-1], self._subgroup_size))]

        return tuple(rngs)

    def __get_workitems_ranges(self):
        return tuple([range(v) for v in self._group_shape])

    @property
    def id(self):
        return self._group_id

    @property
    def shape(self):
        return self._group_shape

    @property
    def size(self):
        if isinstance(self._group_shape, int):
            return self._group_shape

        return prod(self._group_shape)

    @property
    def work_offset(self):
        return tuple(a * b for a, b in zip(self._group_shape, self._group_id))

    def load(self, array, shape, mapping=None):
        if not isinstance(shape, Iterable):
            shape = (shape,)

        dtype = array.dtype
        init = _get_uninit_value(dtype)
        res = _masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=False)
        if mapping:
            f = mapping.func
            src_shape = array.shape
            for index in np.ndindex(*shape):
                mapped = f(*index)
                if all(i >= 0 and i < b for i, b in zip(mapped, src_shape)):
                    res[index] = array[mapped]

        else:
            s = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(array.shape, shape))
            res[s] = array[s]

        return res

    def store(self, dst, src, mapping=None):
        if mapping:
            f = mapping.func
            src_shape = src.shape
            dst_shape = dst.shape
            if isinstance(src, _masked_array):
                for index in np.ndindex(*src_shape):
                    mapped = f(*index)
                    if all(i >= 0 and i < b for i, b in zip(mapped, dst_shape)):
                        val = src[index]
                        if val.mask:
                            dst[mapped] = val.data
            else:
                for index in np.ndindex(*src_shape):
                    mapped = f(*index)
                    if all(i >= 0 and i < b for i, b in zip(mapped, dst_shape)):
                        dst[mapped] = src[index]

            return

        s = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(src.shape, dst.shape))
        if isinstance(src, _masked_array):
            dst[s] = np.where(src.mask[s], src[s], dst[s])
        else:
            dst[s] = src[s]

    def vload(self, array, shape, mapping=None):
        return self.load(array, shape, mapping)

    def empty(self, shape, dtype):
        return self._alloc_impl(shape, dtype, _get_uninit_value(dtype))

    def zeros(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 0)

    def ones(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 1)

    def full(self, shape, dtype, fill_value):
        return self._alloc_impl(shape, dtype, fill_value)

    def vempty(self, shape, dtype):
        return self._alloc_impl(shape, dtype, _get_uninit_value(dtype))

    def vzeros(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 0)

    def vones(self, shape, dtype):
        return self._alloc_impl(shape, dtype, 1)

    def vfull(self, shape, dtype, fill_value):
        return self._alloc_impl(shape, dtype, fill_value)

    def vec(self, src, shape=None):
        if shape is None:
            shape = src.shape

        s = tuple(slice(0, s) for s in shape)
        return src.copy()[s]

    def subgroups(self, func):
        def _body_wrapper(sgid):
            func(CurrentSubGroup(self._subgroup_size, sgid))

        def _func():
            tasks = self._tasks
            assert len(tasks) == 0
            assert self._current_task == 0
            for sgid in product(*self._subgroup_ranges):
                tasks.append(greenlet(partial(_body_wrapper, sgid)))

            for t in tasks:
                t.switch()

            self._current_task = 0
            tasks.clear()

        return _func

    def workitems(self, func):
        def _body_wrapper(lid):
            gid = tuple(
                gi * gs + li
                for gi, gs, li in zip(self._group_id, self._group_shape, lid)
            )
            func(CurrentWorkitem(lid, gid))

        def _func():
            tasks = self._tasks
            assert len(tasks) == 0
            assert self._current_task == 0
            for lid in product(*self._workitem_ranges):
                tasks.append(greenlet(partial(_body_wrapper, lid)))

            for t in tasks:
                t.switch()

            self._current_task = 0
            tasks.clear()

        return _func

    def barrier(self):
        tasks = self._tasks
        num_tasks = len(tasks)
        if num_tasks <= 1:
            return

        next_task = self._current_task + 1
        if next_task >= num_tasks:
            next_task = 0

        self._current_task = next_task
        tasks[next_task].switch()

    def _alloc_impl(self, shape, dtype, init):
        return _masked_array(np.full(shape, fill_value=init, dtype=dtype), mask=True)


class Buffer(typing.Generic[typing.ParamSpec("Args")]):
    pass


def atomic_ref(a):
    return a


class CurrentGroup1(CurrentGroup):
    pass


class CurrentGroup2(CurrentGroup):
    pass


class CurrentGroup3(CurrentGroup):
    pass


def _register_symbol(sym):
    _reg_symbol_impl(eval(sym), sym, __name__)


_register_symbol("Buffer")
_register_symbol("CurrentGroup1")
_register_symbol("CurrentGroup2")
_register_symbol("CurrentGroup3")
_register_symbol("CurrentSubGroup")
_register_symbol("CurrentWorkitem")
