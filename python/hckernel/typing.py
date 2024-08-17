# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_registry import register_symbol as _reg_symbol_impl
from .dispatcher import create_dispatcher
from ._native.compiler._typing import TypingDispatcher, link_modules, load_mlir_module
from .bitcode_storage import get_bitcode_file
from .mlir import ir


def _register_symbol(sym):
    _reg_symbol_impl(eval(sym), sym, __name__)


def _register_func(func):
    _reg_symbol_impl(func, func.__name__, __name__)
    return func


def _stub_error():
    raise NotImplementedError("This is a stub")


_typing_prelink = load_mlir_module(ir._BaseContext.current, get_bitcode_file("typing"))


class TypingRegistry:
    def __init__(self):
        self._typing_dispatchers = []
        self._typing_module = None
        _reg_symbol_impl(self, "TypingRegistry", __name__)

    def add_dispatcher(self, disp):
        self._typing_dispatchers.append(disp)

    def compile_type_resolvers(self):
        typing_module = None
        try:
            for disp in self._typing_dispatchers:
                mod = disp.compile()
                if typing_module is None:
                    typing_module = mod
                else:
                    link_modules(typing_module, mod)

            self._typing_module = typing_module
        finally:
            # To properly handle exceptions in `compile()`
            self._typing_dispatchers.clear()

    @property
    def module(self):
        return self._typing_module


@_register_func
def func(func):
    return create_dispatcher(func, dispatcher=TypingDispatcher)


@_register_func
def type_resolver(registry, key):
    def _wrapper(func):
        disp = create_dispatcher(
            func, prelink_module=_typing_prelink, dispatcher=TypingDispatcher
        )
        registry.add_dispatcher(disp)
        return disp

    return _wrapper


_typing_module = None


def compile_type_resolvers():
    global _typing_dispatchers
    global _typing_module
    try:
        for disp in _typing_dispatchers:
            mod = disp.compile()
            if _typing_module is None:
                _typing_module = mod
            else:
                link_modules(_typing_module, mod)

    finally:
        # To properly handle exceptions in `compile()`
        _typing_dispatchers.clear()


@_register_func
def to_int(a, b):
    _stub_error()


@_register_func
def is_same(a, b):
    _stub_error()


@_register_func
def check(cond):
    _stub_error()


@_register_func
def get_attr(name):
    _stub_error()


@_register_func
def get_num_args():
    _stub_error()


@_register_func
def get_arg(index):
    _stub_error()


@_register_func
def get_named_arg(name):
    _stub_error()


@_register_func
def make_symbol(name):
    _stub_error()


@_register_func
def create_seq():
    _stub_error()


@_register_func
def append_seq(seq, value):
    _stub_error()


@_register_func
def get_seq_element(seq, index):
    _stub_error()


@_register_func
def get_seq_size(seq):
    _stub_error()


@_register_func
def make_type(name, **kwargs):
    _stub_error()


@_register_func
def get_type_name(type):
    _stub_error()


@_register_func
def get_type_param(type, name):
    _stub_error()


@_register_func
def get_global_attr(name):
    _stub_error()
