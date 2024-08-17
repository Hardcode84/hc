# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType
from typing import Callable
from collections import namedtuple, OrderedDict

from .kernel_api import *
from .compiler import mlir_context, Dispatcher
from .symbol_registry import get_module_for_symbol
from .mlir import ir
from .mlir import typing as hc_typing

FuncDesc = namedtuple(
    "FuncDesc",
    [
        "source",
        "name",
        "args",
        "imported_symbols",
        "literals",
        "dispatcher_deps",
        "prelink_module",
        "global_attrs",
    ],
)
from .typename import Typename


def _is_literal(val):
    return isinstance(val, (int, float, ir.Type))


def _process_annotation(ann):
    def istypingtype(a, typ):
        return typing.get_origin(ann) == typ or isinstance(ann, typ)

    def get_typing_args(ann):
        if isinstance(ann, (types.GenericAlias, typing._GenericAlias)):
            return typing.get_args(ann)[0]

        if isinstance(ann, Iterable):
            return ann

        assert False

    if ann in (CurrentGroup, CurrentSubGroup, CurrentWorkitem):
        # nothing
        return

    if isinstance(ann, hc_typing.ValueType):
        # nothing
        return

    if isinstance(ann, Symbol):
        return "sym"

    if isinstance(ann, Typename):
        return "typename"

    elif istypingtype(ann, tuple):
        return tuple(_process_annotation(e) for e in get_typing_args(ann))

    elif istypingtype(ann, Buffer):
        return [_process_annotation(e) for e in get_typing_args(ann)]

    else:
        assert False, f"Unsupported annotation: {type(ann)} {ann}"


def _get_desc(func, dispatcher_cls, prelink_module, global_attrs):
    if not isinstance(func, FunctionType):
        raise RuntimeError(f"Unsupported object {type(func)}")

    def _wrapper():
        if isinstance(prelink_module, Callable):
            prelink_mod = prelink_module()
        else:
            prelink_mod = prelink_module

        sig = inspect.signature(func)
        args_types = OrderedDict()
        for name, param in sig.parameters.items():
            annotation = param.annotation
            if annotation == param.empty:
                continue

            annotation = _process_annotation(annotation)
            if annotation is None:
                continue

            args_types[name] = annotation

        imported_symbols = {}
        literals = {}
        dispatcher_deps = {}
        for name, obj in func.__globals__.items():
            mod = get_module_for_symbol(obj)
            if mod:
                imported_symbols[name] = mod

            if _is_literal(obj):
                literals[name] = obj

            if isinstance(obj, dispatcher_cls):
                dispatcher_deps[name] = obj

        return FuncDesc(
            source=inspect.getsource(func),
            name=func.__name__,
            args=args_types,
            imported_symbols=imported_symbols,
            literals=literals,
            dispatcher_deps=dispatcher_deps,
            prelink_module=prelink_mod,
            global_attrs=global_attrs,
        )

    return _wrapper


def create_dispatcher(
    func, prelink_module=None, dispatcher=Dispatcher, global_attrs=None
):
    return dispatcher(
        mlir_context,
        _get_desc(
            func,
            dispatcher_cls=dispatcher,
            prelink_module=prelink_module,
            global_attrs=global_attrs,
        ),
    )
