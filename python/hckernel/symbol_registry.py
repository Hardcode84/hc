# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


_symbol_registry = {}


def register_symbol(sym, sym_name, module_name, overwrite=False):
    assert module_name is not None
    assert overwrite or (
        sym not in _symbol_registry
    ), f"Symbol is alredy registered: {sym}"
    if isinstance(module_name, str):
        module_name = module_name.split(".")

    module_name.append(sym_name)

    _symbol_registry[sym] = module_name


def get_module_for_symbol(sym):
    try:
        return _symbol_registry.get(sym, None)
    except TypeError:
        return None
