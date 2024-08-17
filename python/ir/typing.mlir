typing.type_resolver ["py_ir.load_module", "hckernel"] {
  %0 = typing.make_ident "hckernel" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.getattr", "typing"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.getattr", "type_resolver"] {
  %c1 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c1
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.type_resolver" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.getattr", "TypingRegistry"] {
  %c1 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c1
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.TypingRegistry" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.getattr", "func"] {
  %c1 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c1
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.func" []
  typing.type_resolver_return %3
}

// typing.to_int

typing.type_resolver ["py_ir.getattr", "to_int"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.to_int" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.to_int" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<index> : !typing.value
  typing.type_resolver_return %3
}

// typing.is_same

typing.type_resolver ["py_ir.getattr", "is_same"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.is_same" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.is_same" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<i1> : !typing.value
  typing.type_resolver_return %3
}

// typing.check

typing.type_resolver ["py_ir.getattr", "check"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.check" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.check" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<none> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_attr

typing.type_resolver ["py_ir.getattr", "get_attr"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_attr" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_attr" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_num_args

typing.type_resolver ["py_ir.getattr", "get_num_args"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_num_args" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_num_args" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<index> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_arg

typing.type_resolver ["py_ir.getattr", "get_arg"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_arg" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_arg" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_named_arg

typing.type_resolver ["py_ir.getattr", "get_named_arg"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_named_arg" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_named_arg" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.make_symbol

typing.type_resolver ["py_ir.getattr", "make_symbol"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.make_symbol" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.make_symbol" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.create_seq

typing.type_resolver ["py_ir.getattr", "create_seq"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.create_seq" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.create_seq" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

typing.type_resolver ["typing.create_seq"] {
  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.append_seq

typing.type_resolver ["py_ir.getattr", "append_seq"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.append_seq" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.append_seq" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_seq_element

typing.type_resolver ["py_ir.getattr", "get_seq_element"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_seq_element" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_seq_element" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_seq_size

typing.type_resolver ["py_ir.getattr", "get_seq_size"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_seq_size" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_seq_size" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<index> : !typing.value
  typing.type_resolver_return %3
}

// typing.make_ident

typing.type_resolver ["py_ir.getattr", "make_type"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.make_type" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.make_type" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_ident_name

typing.type_resolver ["py_ir.getattr", "get_type_name"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_type_name" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_type_name" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_ident_param

typing.type_resolver ["py_ir.getattr", "get_type_param"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_type_param" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_type_param" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

typing.type_resolver ["typing.get_ident_param"] {
  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// typing.get_global_attr

typing.type_resolver ["py_ir.getattr", "get_global_attr"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.get_global_attr" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.get_global_attr" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.type_constant #typing.type_attr<!typing.value> : !typing.value
  typing.type_resolver_return %3
}

// join types

typing.type_resolver "join_types" {
  %0 = typing.type_constant #typing.type_attr<index> : !typing.value
  %c0 = arith.constant 0: index
  %1 = typing.get_arg %c0

  %c1 = arith.constant 1: index
  %2 = typing.get_arg %c1

  %3 = typing.is_same %0 %1
  %4 = typing.is_same %0 %2
  %5 = arith.ori %3, %4 : i1
  typing.check %5

  typing.type_resolver_return %0
}

// py_ir ops

typing.type_resolver ["py_ir.binop"] {
  %0 = typing.type_constant #typing.type_attr<i1> : !typing.value
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %a0 = typing.get_arg %c0
  %a1 = typing.get_arg %c1
  %i0 = typing.is_same %a0 %0
  %i1 = typing.is_same %a1 %0
  typing.check %i0
  typing.check %i1

  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.inplace_binop"] {
  %0 = typing.type_constant #typing.type_attr<i1> : !typing.value
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %a0 = typing.get_arg %c0
  %a1 = typing.get_arg %c1
  %i0 = typing.is_same %a0 %0
  %i1 = typing.is_same %a1 %0
  typing.check %i0
  typing.check %i1

  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.binop"] {
  %0 = typing.type_constant #typing.type_attr<index> : !typing.value
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.inplace_binop"] {
  %0 = typing.type_constant #typing.type_attr<index> : !typing.value
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.make_list"] {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = typing.get_num_args
  %seq = typing.create_seq
  cf.br ^bb1(%c0, %seq : index, !typing.value)
^bb1(%idx: index, %s1: !typing.value):
  %cond = arith.cmpi slt, %idx, %n : index
  %next = arith.addi %idx, %c1 : index
  cf.cond_br %cond, ^bb2, ^bb3(%s1: !typing.value)
^bb2:
  %arg = typing.get_arg %idx
  %new_seq = typing.append_seq %s1 %arg
  cf.br ^bb1(%next, %new_seq : index, !typing.value)
^bb3(%s2: !typing.value):
  %0 = typing.make_ident "List" ["Elements"] : %s2
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.type_resolver" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %c1 = arith.constant 1: index
  %3 = typing.get_arg %c1
  %4 = typing.get_ident_name %3
  %5 = typing.make_literal "hckernel.typing.TypingRegistry"
  %6 = typing.is_same %4 %5
  typing.check %6

  %c2 = arith.constant 2: index
  %7 = typing.get_arg %c2
  %8 = typing.get_ident_name %7
  %9 = typing.make_literal "List"
  %10 = typing.is_same %8 %9
  typing.check %10

  %11 = typing.get_ident_param %7 "Elements"
  %12 = typing.make_ident "type_resolver_type" ["key"] : %11
  typing.type_resolver_return %12
}
