import sympy as sp

# Input coords and general functions.
dim = int(input("How many coords?: "))
coord_names = input(f"Enter {dim} coords separated by spaces: ").split()
coords = sp.symbols(coord_names)
coord_dict = dict(zip(coord_names, coords))
genfuncs = {}
isfunc = input("Are there any general functions? (y/n): ")
if isfunc.lower() == 'y':
    nfuncs = int(input("How many general functions?: "))
    for i in range(nfuncs):
        parts = input(f"Enter function #{i+1} name and its variables separated by spaces: ").split()
        func_name = parts[0]
        vars_names = parts[1:]
        vars_syms = [coord_dict[v] for v in vars_names]
        genfuncs[func_name] = sp.Function(func_name)(*vars_syms)
print()

# Input metric.
print(f"Enter {dim}x{dim} metric components row by row, separated by spaces:")
g_list = []
local_env = {**coord_dict, **genfuncs}
for i in range(dim):
    row = input(f"Row {i+1}: ").split()
    g_list.append([sp.sympify(x, locals=local_env) for x in row])
g = sp.Matrix(g_list)
g_inv = g.inv()
print()

# Compute Christoffel tensor components.
Γ = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            Γ[i, j, k] = sp.simplify(sp.Rational(1, 2) * sum(
                g_inv[i, m] * (
                    sp.diff(g[m, j], coords[k]) +
                    sp.diff(g[m, k], coords[j]) -
                    sp.diff(g[j, k], coords[m])
                )
                for m in range(dim)
            ))

# Output nonzero Christoffel symbols with symmetric pairs.
printed = set()
print("\nNonzero Christoffel Symbols:")
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            val = Γ[i,j,k]
            if val != 0:
                if (i,k,j) not in printed:
                    printed.add((i,j,k))
                    printed.add((i,k,j))
                    if j != k:
                        print(f"Γ({coords[i]}_{coords[j]}{coords[k]}) = Γ({coords[i]}_{coords[k]}{coords[j]}) =", val)
                    else:
                        print(f"Γ({coords[i]}_{coords[j]}{coords[k]}) =", val)
print()

# D represents differentiation wrt arbitrary paramater λ.
D_symbols = sp.symbols([f'D{coord}' for coord in coord_names])
D = dict(zip(coords, D_symbols))
D2_symbols = sp.symbols([f'D2{coord}' for coord in coord_names])
D2 = dict(zip(coords, D2_symbols))

# Compute geodesic equations.
print('Geodesic Equations:')
geodesic = {}
for i in range(dim):
    geodesic[coords[i]] = D2[coords[i]] + sum(
        Γ[i, j, k] * D[coords[j]] * D[coords[k]]
        for j in range(dim)
        for k in range(dim)
    )
    print(f'{coords[i]}: 0 = {geodesic[coords[i]]}')
print()

# Identify cyclic coordinates and compute conjugate momenta.
p = {}
for i in range(dim):
    if not g.has(coords[i]):
        print(f'{coords[i]} is cyclic and its conjugate momentum is conserved:')
        p[coords[i]] = sp.simplify(sum(g[i, j] * D[coords[j]] for j in range(dim)))
        print(f'p_{coords[i]} =', p[coords[i]])
print()

# Compute once contravariant Riemann curvature tensor.
print('Nonzero Riemann Curvature Components:')
Rie = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            for l in range(dim):
                Rie[i,j,k,l] = sp.simplify(
                    sp.diff(Γ[i, j, l], coords[k]) -
                    sp.diff(Γ[i, j, k], coords[l]) +
                    sum(
                        Γ[i, m, k] * Γ[m, j, l] -
                        Γ[i, m, l] * Γ[m, j, k]
                        for m in range(dim)
                    )
                )
                if Rie[i,j,k,l] != 0:
                    print(f'Rie({coords[i]}{coords[j]}{coords[k]}{coords[l]}) = ', Rie[i,j,k,l])
print()

# Compute Ricci curvature tensor and scalar.
print('Nonzero Ricci Curvature Components:')
Ric = sp.MutableDenseNDimArray.zeros(dim, dim)
for i in range(dim):
    for j in range(dim):
        Ric[i,j] = sp.simplify(sum(Rie[m,i,m,j] for m in range(dim)))
        if Ric[i,j] != 0:
            print(f'Ric({coords[i]}, {coords[j]}) = ', Ric[i,j])
print('The Ricci scalar is: ')
R = sp.simplify(sum(g_inv[i,j] * Ric[i,j] for i in range(dim) for j in range(dim)))
print('R = ', R)
print()

