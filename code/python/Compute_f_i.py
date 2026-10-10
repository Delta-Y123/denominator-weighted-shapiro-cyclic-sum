import sympy as sp

def main():
    U, V, p, q = sp.symbols('U V p q', real=True, positive=True)
    u, v = sp.symbols('u v', real=True)

    def compute_f(i):
        f = {}
        f[-1] = U
        f[0] = 0
        f[1] = 1
        f[2] = q**2 * (V - 1) / (p**2 * U**2)
        for n in range(3, i + 1):
            A = q**2 * f[n-4] / (p * f[n-3] + q * f[n-2])**2
            B = p * q * f[n-3] / (p * f[n-2] + q * f[n-1])**2
            expr = 1 / (A + B) - (p / q) * f[n-1]
            expr = sp.cancel(expr)
            f[n] = expr
        return f[i]

    try:
        i = int(input("Enter i (≥ -1): "))
        if i < -1:
            print("i must be ≥ -1")
            return
    except ValueError:
        print("Please enter an integer")
        return

    expr_U = compute_f(i)
    expr_uv = expr_U.subs({U: u, V: v})

    print(f"\nExpression for f_{i}^l (parameters p, q retained):")
    sp.pprint(expr_uv, use_unicode=True)
    print("\nLaTeX format:")
    print(sp.latex(expr_uv))

    response = input("\nSubstitute a numerical value for p? (y/n, default n): ").strip().lower()
    if response == 'y':
        p_input = input("Enter value for p (q = 1 - p, supports fractions/decimals/radicals): ")
        try:
            p_val = sp.sympify(p_input)
            if p_val.is_number:
                q_val = 1 - p_val
                expr_sub = expr_uv.subs({p: p_val, q: q_val})
                expr_sub = sp.cancel(expr_sub)
                print(f"\nAfter substituting p = {p_val}, q = {q_val}, f_{i}^l becomes:")
                sp.pprint(expr_sub, use_unicode=True)
                print("\nLaTeX format:")
                print(sp.latex(expr_sub))
            else:
                print("Invalid numerical value. Please run the program again and enter a number.")
        except Exception as e:
            print(f"Input processing error: {e}")

if __name__ == "__main__":
    main()