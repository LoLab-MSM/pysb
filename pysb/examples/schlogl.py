from pysb import *

Model()
Parameter('atol', 1e5)
Parameter('Btot', 2e5)
Parameter('k1', 3e-7)
Parameter('k2', 1e-4)
Parameter('k2_n', 1e-4/3)
Parameter('k3', 1e-3)
Parameter('k4', 3.5)
Parameter('atol_k1', 1e5 * 3e-7)
Parameter('btol_k3', 2e5 * 1e-3)
Monomer('X')
Monomer('A')
Monomer('B')
Monomer('I')
Initial(X(), Parameter('X_0', 100))
Initial(A(), Parameter('A_0', 1e5))
Initial(B(), Parameter('B_0', 1e5))
Initial(I(), Parameter('I_0', 1))

#Rule('rule1', A() + X() + X() <> A() + X() + X() + X(), k1, k2)
#Rule('rule2', B() <> X() + B(), k3, k4)

Rule('rule3', X() + X() <> X() + X() + X(), atol_k1, k2_n)
Rule('rule4', I() <> X() + I() , btol_k3, k4)
Observable('X_total', X())
Observable('A_total', A())
Observable('B_total', B())
