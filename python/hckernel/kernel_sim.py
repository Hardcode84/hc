# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .kernel_api import _divup, _verify_kernel_params
from .kernel_api import *

DEF_GROUP_SHAPE = 64
DEF_SUBGROUP_SIZE = 16


def _get_num_groups(work_shape, group_shape):
    return tuple(_divup(a, b) for a, b in zip(work_shape, group_shape))


def _get_default_tunables(tunables):
    if isinstance(tunables, TunableParam):
        return {tunables.sym: tunables.default}

    if isinstance(tunables, Iterable):
        return {t.sym: t.default for t in tunables}

    assert False, "Unsupported tunables"


def _get_dims(arg):
    if not isinstance(arg, Iterable):
        return (arg,)

    return arg


def _add_sub(subs, sym, val):
    if sym in subs:
        old_val = subs[sym]
        assert old_val == val, f"Symbol value conflict for {sym}: {old_val} and {val}"
    else:
        subs[sym] = val


def _visit_arg_annotation(idx, ann, prev_handler):
    def istypingtype(a, typ):
        return typing.get_origin(ann) == typ or isinstance(ann, typ)

    def get_typing_args(ann):
        if isinstance(ann, (types.GenericAlias, typing._GenericAlias)):
            return typing.get_args(ann)

        if isinstance(ann, Iterable):
            return ann

        assert False

    if ann in (CurrentGroup, CurrentSubGroup, CurrentWorkitem):
        # nothing
        return

    if isinstance(ann, Symbol):

        def handler(subs, args):
            val = args[idx]
            _add_sub(subs, ann, val)

    elif istypingtype(ann, tuple):

        def handler(subs, args):
            val = args[idx]
            assert isinstance(val, Iterable)
            ann_args = get_typing_args(ann)
            assert len(val) == len(ann_args)
            for s, v in zip(ann_args, val):
                if not isinstance(s, Symbol):
                    continue

                _add_sub(subs, s, v)

    elif istypingtype(ann, Buffer):

        def handler(subs, args):
            val = args[idx]
            ann_args = typing.get_args(ann)[0]
            shape = ann_args[:-1]
            dtype = ann_args[-1]
            assert len(val.shape) == len(shape)
            for s, v in zip(shape, val.shape):
                if not isinstance(s, Symbol):
                    continue

                _add_sub(subs, s, v)

    else:
        assert False, f"Unsupported annotation: {type(ann)} {ann}"

    if prev_handler:

        def chained(subs, args):
            prev_handler(subs, args)
            handler(subs, args)

        return chained
    else:
        return handler


def _visit_func_annotations(func):
    ann = inspect.get_annotations(func)
    handler = None
    for i, arg in enumerate(inspect.signature(func).parameters):
        arg_ann = ann.get(arg, None)
        if arg_ann is None:
            continue

        handler = _visit_arg_annotation(i - 1, arg_ann, handler)

    return handler


def _handle_dim(src, subs):
    if isinstance(src, Iterable):
        return tuple(_handle_dim(v, subs) for v in src)

    if isinstance(src, Expr):
        src = src.subs(subs)

    return int(src)


class _kernel:
    def __init__(
        self, func, work_shape, group_shape, subgroup_size, literals, tunables, tuned={}
    ):
        self.handler = _visit_func_annotations(func)
        self.orig_func = func
        self.work_shape = work_shape
        self.group_shape = group_shape
        self.subgroup_size = subgroup_size
        self.literals = literals
        self.tunables = tunables
        self.tuned = tuned
        self.default_tunbles = _get_default_tunables(tunables)

    def __call__(self, *args, **kwargs):
        if self.handler:
            subs_map = {}
            self.handler(subs_map, args)
            subs_map = subs_map | self.default_tunbles | self.tuned
            subs_args = list(subs_map.items())
        else:
            subs_args = []

        ws = _handle_dim(self.work_shape, subs_args)
        gs = _handle_dim(self.group_shape, subs_args)
        ss = _handle_dim(self.subgroup_size, subs_args)

        resolved_func = resolve_symbols(self.orig_func, subs_map)

        n_groups = _get_num_groups(ws, gs)
        cg = CurrentGroup(gs, ss)
        for gid in product(*(range(g) for g in n_groups)):
            cg._group_id = gid
            resolved_func(cg, *args, **kwargs)

    def parametrize(self, tuned):
        return _kernel(
            self.orig_func,
            self.work_shape,
            self.group_shape,
            self.subgroup_size,
            self.literals,
            self.tunables,
            self.tuned | tuned,
        )


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
):
    _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables)
    if subgroup_size is None:
        subgroup_size = DEF_SUBGROUP_SIZE

    work_shape = _get_dims(work_shape)

    if group_shape is None:
        group_shape = (DEF_GROUP_SHAPE,)
        while len(group_shape) < len(work_shape):
            group_shape = (1,) + group_shape

    group_shape = _get_dims(group_shape)

    assert len(work_shape) <= 3
    assert len(work_shape) == len(group_shape)

    def _kernel_impl(func):
        return _kernel(func, work_shape, group_shape, subgroup_size, literals, tunables)

    return _kernel_impl
