import numpy as np
import matplotlib.pyplot as plt

def plot_region(p):
    q = 1 - p
    if not 0 < p < 1:
        print("Please provide a value with 0 < p < 1 to plot the region.")
        return

    u_min = q
    u_max = (q / p) * (q / p)
    v_max = 1 + (q**4 / p**4) / 4

    if u_max < u_min:
        print("Cannot plot region because the maximum of u is not greater than the minimum. Please check the value of p.")
        return

    num_points = 500
    u_vec = np.linspace(u_min, u_max, num_points)
    v_vec = np.linspace(1, v_max, num_points)

    U_mesh, V_mesh = np.meshgrid(u_vec, v_vec)

    parabola_values = 1 + (p / q) * U_mesh * (1 - (p**2 / q**2) * U_mesh)

    region_mask = (V_mesh > 1) & (V_mesh <= parabola_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    mask_valid_u = (u_vec >= 0) & (u_vec <= q**2 / p**2 + 1)
    u_parabola = u_vec[mask_valid_u]
    v_parabola_upper = 1 + (p / q) * u_parabola - (p**3 / q**3) * u_parabola**2
    v_parabola_lower = np.ones_like(u_parabola)

    u_polygon = np.concatenate([u_parabola, u_parabola[::-1]])
    v_polygon = np.concatenate([v_parabola_upper, v_parabola_lower[::-1]])

    ax.fill(u_polygon, v_polygon, color='skyblue', alpha=0.7, label=f'$(u,v)\\in UV_{{{p:.2f}}} $')

    ax.set_title(f"$1 \\leq v \\leq 1 + \\frac{{p}}{{q}}u - \\frac{{p^3}}{{q^3}}u^2$\n$p = {p:.2f}, q = {q:.2f}$\n$1 \\leq v \\leq 1 + {p/q:.2f}u - {p**3 / q**3:.2f}u^2$")
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.axis('equal')
    ax.set_xlim(u_min, u_max)
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        p_input = float(input("Enter the value of p (0 < p < 1): "))
        plot_region(p_input)
    except ValueError as e:
        print(f"Input error: {e}")