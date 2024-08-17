# RUN: pyfront front %s | FileCheck %s


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: py_ast.pass
def func():
    pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: %[[A1:.*]] = py_ast.arg "a"
#       CHECK: %[[A2:.*]] = py_ast.arg "b"
#       CHECK: %[[A3:.*]] = py_ast.arg "c"
#       CHECK: py_ast.func "func"(%[[A1]], %[[A2]], %[[A3]])
#       CHECK: py_ast.pass
def func(a, b, c):
    pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: %[[A:.*]] = py_ast.name "Foo"
#       CHECK: %[[ARG:.*]] = py_ast.arg "a" : %[[A]]
#       CHECK: py_ast.func "func"(%[[ARG]])
#       CHECK: py_ast.pass
def func(a: Foo):
    pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: %[[F:.*]] = py_ast.name "Foo"
#       CHECK: %[[B:.*]] = py_ast.name "Bar"
#       CHECK: %[[S:.*]] = py_ast.subscript %[[F]][%[[B]]]
#       CHECK: %3 = py_ast.arg "a" : %[[S]]
def func(a: Foo[Bar]):
    pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[F:.*]] = py_ast.name "Foo"
#       CHECK: py_ast.expr %[[F]]
def func():
    Foo


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A1:.*]] = py_ast.name "Foo"
#       CHECK: %[[A2:.*]] = py_ast.name "Bar"
#       CHECK: %[[A3:.*]] = py_ast.name "Baz"
#       CHECK: %[[T:.*]] = py_ast.tuple %[[A1]], %[[A2]], %[[A3]]
#       CHECK: py_ast.expr %[[T]]
def func():
    Foo, Bar, Baz


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A1:.*]] = py_ast.name "Foo"
#       CHECK: %[[A2:.*]] = py_ast.name "Bar"
#       CHECK: %[[A3:.*]] = py_ast.name "Baz"
#       CHECK: %[[T:.*]] = py_ast.list %[[A1]], %[[A2]], %[[A3]]
#       CHECK: py_ast.expr %[[T]]
def func():
    [Foo, Bar, Baz]


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[F:.*]] = py_ast.name "Foo"
#       CHECK: %[[A:.*]] = py_ast.attribute %[[F]] attr "Bar"
#       CHECK: py_ast.expr %[[A]]
def func():
    Foo.Bar


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[C1:.*]] = py_ast.constant 1 : i64
#       CHECK: py_ast.expr %[[C1]]
#       CHECK: %[[C2:.*]] = py_ast.constant 255 : i64
#       CHECK: py_ast.expr %[[C2]]
#       CHECK: %[[C3:.*]] = py_ast.constant 2.5{{.*}}e+00 : f64
#       CHECK: py_ast.expr %[[C3]]
#       CHECK: %[[C4:.*]] = py_ast.constant #complex.number<:f64 0.{{.*}}e+00, 3.{{.*}}e+00> : complex<f64>
#       CHECK: py_ast.expr %[[C4]]
#       CHECK: %[[C5:.*]] = py_ast.constant "Test"
#       CHECK: py_ast.expr %[[C5]]
#       CHECK: %[[C6:.*]] = py_ast.constant #py_ast.none
#       CHECK: py_ast.expr %[[C6]]
def func():
    1
    0xFF
    2.5
    3j
    "Test"
    None


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[I1:.*]] = py_ast.constant 1 : i64
#       CHECK: %[[I2:.*]] = py_ast.constant 2 : i64
#       CHECK: %[[I3:.*]] = py_ast.constant 3 : i64
#       CHECK: %[[S:.*]] = py_ast.slice(%[[I1]] : %[[I2]] : %[[I3]])
#       CHECK: %[[SU:.*]] = py_ast.subscript %[[A]][%[[S]]]
#       CHECK: py_ast.expr %[[SU]]
def func():
    A[1:2:3]


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[F:.*]] = py_ast.name "Foo"
#       CHECK: %[[B:.*]] = py_ast.name "Bar"
#       CHECK: py_ast.assign(%[[F]]) = %[[B]]
def func():
    Foo = Bar


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[F:.*]] = py_ast.name "Foo"
#       CHECK: %[[B:.*]] = py_ast.name "Bar"
#       CHECK: %[[Z:.*]] = py_ast.name "Baz"
#       CHECK: py_ast.assign(%[[F]], %[[B]]) = %[[Z]]
def func():
    Foo = Bar = Baz


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK:  %[[A:.*]] = py_ast.name "A"
#       CHECK:  %[[B:.*]] = py_ast.name "B"
#       CHECK:  %[[T:.*]] = py_ast.tuple %[[A]], %[[B]]
#       CHECK:  %[[C:.*]] = py_ast.name "C"
#       CHECK:  %[[D:.*]] = py_ast.name "D"
#       CHECK:  %[[R:.*]] = py_ast.tuple %[[C]], %[[D]]
#       CHECK: py_ast.assign(%[[T]]) = %[[R]]
def func():
    A, B = C, D


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "a"
#       CHECK: %[[B:.*]] = py_ast.name "b"
#       CHECK: %[[F:.*]] = py_ast.name "foo"
#       CHECK: %[[R:.*]] = py_ast.call %[[F]](%[[A]], %[[B]] keywords )
#       CHECK: py_ast.expr %[[R]]
def func():
    foo(a, b)


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "a"
#       CHECK: %[[B:.*]] = py_ast.name "b"
#       CHECK: %[[BB:.*]] = py_ast.keyword "B" = %[[B]]
#       CHECK: %[[F:.*]] = py_ast.name "foo"
#       CHECK: %[[R:.*]] = py_ast.call %[[F]](%[[A]] keywords %[[BB]])
#       CHECK: py_ast.expr %[[R]]
def func():
    foo(a, B=b)


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "a"
#       CHECK: %[[AA:.*]] = py_ast.keyword "A" = %[[A]]
#       CHECK: %[[B:.*]] = py_ast.name "b"
#       CHECK: %[[BB:.*]] = py_ast.keyword "B" = %[[B]]
#       CHECK: %[[F:.*]] = py_ast.name "foo"
#       CHECK: %[[R:.*]] = py_ast.call %[[F]]( keywords %[[AA]], %[[BB]])
#       CHECK: py_ast.expr %[[R]]
def func():
    foo(A=a, B=b)


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: %[[R:.*]] = py_ast.bool_op and, %[[A]], %[[B]]
#       CHECK: py_ast.expr %[[R]]
def func():
    A and B


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: %[[C:.*]] = py_ast.name "C"
#       CHECK: %[[R:.*]] = py_ast.bool_op or, %[[A]], %[[B]], %[[C]]
#       CHECK: py_ast.expr %[[R]]
def func():
    A or B or C


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[T:.*]] = py_ast.name "A"
#       CHECK: py_ast.if %[[T]] {
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.expr %[[B]]
#       CHECK: }
def func():
    if A:
        B


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[T:.*]] = py_ast.name "A"
#       CHECK: py_ast.if %[[T]] {
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.expr %[[B]]
#       CHECK: } {
#       CHECK: %[[C:.*]] = py_ast.name "C"
#       CHECK: py_ast.expr %[[C]]
#       CHECK: }
def func():
    if A:
        B
    else:
        C


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[T:.*]] = py_ast.name "A"
#       CHECK: py_ast.if %[[T]] {
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.expr %[[B]]
#       CHECK: } {
#       CHECK: %[[C:.*]] = py_ast.name "C"
#       CHECK: py_ast.if %[[C]] {
#       CHECK: %[[D:.*]] = py_ast.name "D"
#       CHECK: py_ast.expr %[[D]]
#       CHECK: } {
#       CHECK: %[[E:.*]] = py_ast.name "E"
#       CHECK: py_ast.expr %[[E]]
#       CHECK: }
def func():
    if A:
        B
    elif C:
        D
    else:
        E


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: %[[A1:.*]] = py_ast.arg "a"
#       CHECK: %[[A2:.*]] = py_ast.arg "b"
#       CHECK: %[[F:.*]] = py_ast.name "foo"
#       CHECK: %[[B:.*]] = py_ast.name "bar"
#       CHECK: py_ast.func "func"(%[[A1]], %[[A2]]) decorators %[[F]], %[[B]]
#       CHECK: py_ast.pass
@foo
@bar
def func(a, b):
    pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: %[[R:.*]] = py_ast.compare %[[A]] [0] %[[B]]
#       CHECK: py_ast.expr %[[R]]
def func():
    A == B


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: %[[C:.*]] = py_ast.name "C"
#       CHECK: %[[R:.*]] = py_ast.compare %[[A]] [2, 4] %[[B]], %[[C]]
#       CHECK: py_ast.expr %[[R]]
def func():
    A < B > C


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[B:.*]] = py_ast.name "B"
#       CHECK: %[[R:.*]] = py_ast.binop %[[A]] add %[[B]]
#       CHECK: py_ast.expr %[[R]]
def func():
    A + B


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: py_ast.return
def func():
    return


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[R:.*]] = py_ast.constant #py_ast.none
#       CHECK: py_ast.return %[[R]]
def func():
    return None


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK-DAG: %[[A:.*]] = py_ast.name "A"
#       CHECK-DAG: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.aug_assign %[[A]] add %[[B]]
def func():
    A += B


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK-DAG: %[[A:.*]] = py_ast.name "A"
#       CHECK-DAG: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.for %[[A]] in %[[B]]
#       CHECK: py_ast.pass
def func():
    for A in B:
        pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK-DAG: %[[A:.*]] = py_ast.name "A"
#       CHECK-DAG: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.for %[[A]] in %[[B]]
#       CHECK: py_ast.break
def func():
    for A in B:
        break


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK-DAG: %[[A:.*]] = py_ast.name "A"
#       CHECK-DAG: %[[B:.*]] = py_ast.name "B"
#       CHECK: py_ast.for %[[A]] in %[[B]]
#       CHECK: py_ast.continue
def func():
    for A in B:
        continue


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func "func"()
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: py_ast.while %[[A]]
#       CHECK: py_ast.pass
def func():
    while A:
        pass


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func
#       CHECK: %[[A:.*]] = py_ast.name "A"
#       CHECK: %[[OpR:.*]] = py_ast.unaryop unot %[[A]]
#       CHECK: py_ast.return %[[OpR]]
def func():
    return not A


# -----


# CHECK-LABEL: py_ast.module
#       CHECK: py_ast.func
#       CHECK-DAG: %[[A:.*]] = py_ast.name "A"
#       CHECK-DAG: %[[B:.*]] = py_ast.name "B"
#       CHECK-DAG: %[[C:.*]] = py_ast.name "C"
#       CHECK: %[[OpR:.*]] = py_ast.ifexp %[[A]] if %[[B]] else %[[C]]
#       CHECK: py_ast.return %[[OpR]]
def func():
    return A if B else C
