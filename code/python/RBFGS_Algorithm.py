import numpy as np
import time
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from datetime import datetime
import os

plt.rcParams['font.family'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def safe_norm(x: np.ndarray, eps: float = 1e-10) -> float:
    norm_val = np.linalg.norm(x)
    return norm_val if norm_val > eps else eps

def safe_pow(x: np.ndarray, power: float, eps: float = 1e-10, min_val: float = 1e-8) -> np.ndarray:
    x = np.where(np.abs(x) < eps, min_val, x)
    return np.power(x, power)

class SphereManifold:
    def __init__(self, n: int):
        self.n = n
        if n < 2:
            raise ValueError("Dimension n must be ≥ 2")
        self.is_even = n % 2 == 0

    def proj(self, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        return v - np.dot(y, v) * y

    def retraction(self, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
        y_new = y + eta
        return y_new / safe_norm(y_new)

    def weighted_shapiro_objective(self, y: np.ndarray, p: float) -> float:
        q = 1.0 - p
        x = y ** 2
        n = self.n
        
        obj_sum = 0.0
        for i in range(n):
            i_next1 = (i + 1) % n
            i_next2 = (i + 2) % n
            
            numerator = x[i]
            denominator = p * x[i_next1] + q * x[i_next2]
            
            denominator = max(denominator, 1e-10)
            
            obj_sum += numerator / denominator
        
        return obj_sum

    def gradient(self, y: np.ndarray, p: float) -> np.ndarray:
        q = 1.0 - p
        x = y ** 2
        n = self.n
        
        egrad = np.zeros(n)
        
        for i in range(n):
            i_next1 = (i + 1) % n
            i_next2 = (i + 2) % n
            denom_i = p * x[i_next1] + q * x[i_next2]
            denom_i = max(denom_i, 1e-10)
            
            egrad[i] += 2 * y[i] / denom_i
            
            i_prev1 = (i - 1) % n
            i_next1_from_prev = i
            i_next2_from_prev = (i + 1) % n
            numer_prev1 = x[i_prev1]
            denom_prev1 = p * x[i] + q * x[i_next2_from_prev]
            denom_prev1 = max(denom_prev1, 1e-10)
            
            egrad[i] -= numer_prev1 * p * 2 * y[i] / (denom_prev1 ** 2)
            
            i_prev2 = (i - 2) % n
            i_prev1_from_prev2 = (i - 1) % n
            numer_prev2 = x[i_prev2]
            denom_prev2 = p * x[i_prev1_from_prev2] + q * x[i]
            denom_prev2 = max(denom_prev2, 1e-10)
            
            egrad[i] -= numer_prev2 * q * 2 * y[i] / (denom_prev2 ** 2)
        
        grad = self.proj(y, egrad)
        
        grad = np.clip(grad, -1e3, 1e3)
        
        return grad

    def parallel_transport(self, y: np.ndarray, eta: np.ndarray, xi: np.ndarray) -> np.ndarray:
        y_new = self.retraction(y, eta)
        y_dot_ynew = np.dot(y, y_new)
        denominator = 1 + y_dot_ynew
        
        if abs(denominator) < 1e-10:
            return self.proj(y_new, xi)
        
        transport_term = np.dot(y + y_new, xi) / denominator * (y + y_new)
        return xi - transport_term
    
    def is_alternating_pattern(self, y: np.ndarray, tol: float = 0.01) -> Tuple[bool, float, float]:
        if not self.is_even:
            return False, 0.0, 0.0
        
        n = self.n
        odd_values = y[0:n:2]
        even_values = y[1:n:2]
        
        odd_std = np.std(odd_values)
        even_std = np.std(even_values)
        
        odd_range = np.max(odd_values) - np.min(odd_values)
        even_range = np.max(even_values) - np.min(even_values)
        
        odd_mean = np.mean(odd_values)
        even_mean = np.mean(even_values)
        
        odd_uniform = odd_std / (abs(odd_mean) + 1e-10) < tol
        even_uniform = even_std / (abs(even_mean) + 1e-10) < tol
        
        if abs(odd_mean) > 1e-10 and abs(even_mean) > 1e-10:
            relative_diff = abs(odd_mean - even_mean) / max(abs(odd_mean), abs(even_mean))
        else:
            relative_diff = abs(odd_mean - even_mean)
        
        is_alternating = odd_uniform and even_uniform and relative_diff > 0.1
        
        return is_alternating, odd_mean, even_mean
    
    def perturb_alternating_solution(self, y: np.ndarray, perturbation_factor: float = 0.05) -> np.ndarray:
        n = len(y)
        perturbation = np.random.randn(n) * perturbation_factor
        y_perturbed = y + perturbation
        
        y_perturbed = y_perturbed / safe_norm(y_perturbed)
        
        return y_perturbed
    
class WeightedShapiroRBFGS:
    def __init__(self, manifold: SphereManifold, p: float):
        self.manifold = manifold
        self.p = p
        self.n = manifold.n

    def _objective(self, y: np.ndarray) -> float:
        return self.manifold.weighted_shapiro_objective(y, self.p)

    def _line_search_armijo(self, y: np.ndarray, eta: np.ndarray, grad: np.ndarray) -> float:
        c1 = 1e-4
        alpha = 1.0
        max_ls_iter = 20
        f_y = self._objective(y)
        grad_dot_eta = np.dot(grad, eta)

        if grad_dot_eta > -1e-10:
            eta = -eta
            grad_dot_eta = -grad_dot_eta

        for _ in range(max_ls_iter):
            y_new = self.manifold.retraction(y, alpha * eta)
            f_y_new = self._objective(y_new)
            if f_y_new <= f_y + c1 * alpha * grad_dot_eta:
                return alpha
            alpha *= 0.5

        return alpha

    def _bfgs_update(self, H: np.ndarray, s: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        yTs = np.dot(y_vec, s)
        if yTs < 1e-10:
            return np.eye(self.n)

        I = np.eye(self.n)
        syT = np.outer(s, y_vec)
        ysT = np.outer(y_vec, s)
        ssT = np.outer(s, s)

        term1 = I - syT / yTs
        term2 = I - ysT / yTs
        term3 = ssT / yTs

        H_new = term1 @ H @ term2 + term3
        return H_new

    def optimize(self, y0: Optional[np.ndarray] = None, max_iter: int = 1000,
                 tol: float = 1e-6, max_bfgs_iter: int = 1000, 
                 restart_on_alternating: bool = True) -> Dict:
        if y0 is None:
            y0 = np.random.randn(self.n)
            y0 = y0 / safe_norm(y0)
        else:
            y0 = y0 / safe_norm(y0)

        y = y0.copy()
        H = np.eye(self.n)
        precision = 1e-4
        
        grad_norm_history = []
        obj_history = []
        start_time = time.time()
        
        prev_grad = None
        prev_eta = None
        prev_y = None
        
        alternating_detected = False
        alternating_info = None

        for k in range(max_iter):
            grad = self.manifold.gradient(y, self.p)
            grad_norm = np.linalg.norm(grad)
            grad_norm_history.append(grad_norm)
            
            obj_val = self._objective(y)
            obj_history.append(obj_val)
            
            if (self.n % 2 == 0 and restart_on_alternating and 
                k > 50 and grad_norm < 1e-2):
                is_alternating, a_val, b_val = self.manifold.is_alternating_pattern(y)
                if is_alternating:
                    alternating_detected = True
                    alternating_info = {
                        'a': a_val,
                        'b': b_val,
                        'iteration': k,
                        'objective': obj_val
                    }
                    print(f"  Alternating pattern detected: a={a_val:.4f}, b={b_val:.4f}, objective={obj_val:.6f}")
                    
                    if obj_val >= self.n - 1e-6:
                        print(f"  Alternating pattern satisfies inequality, early termination")
                        break

            if grad_norm < tol:
                break

            if k == 0:
                eta = -grad
            else:
                eta = -H @ grad

            alpha = self._line_search_armijo(y, eta, grad)

            y_new = self.manifold.retraction(y, alpha * eta)

            if k > 0 and k < max_bfgs_iter:
                s_transported = self.manifold.parallel_transport(prev_y, alpha * prev_eta, alpha * prev_eta)
                grad_transported = self.manifold.parallel_transport(prev_y, alpha * prev_eta, prev_grad)
                grad_new = self.manifold.gradient(y_new, self.p)
                y_vec = grad_new - grad_transported

                H = self._bfgs_update(H, s_transported, y_vec)

            prev_grad = grad.copy()
            prev_eta = eta.copy()
            prev_y = y.copy()
            y = y_new

        end_time = time.time()
        final_grad = self.manifold.gradient(y, self.p)
        final_grad_norm = np.linalg.norm(final_grad)
        final_obj = self._objective(y)

        return {
            "y_opt": y,
            "x_opt": y ** 2,
            "obj_opt": final_obj,
            "iter_num": k + 1,
            "compute_time": end_time - start_time,
            "grad_norm_history": np.array(grad_norm_history),
            "obj_history": np.array(obj_history),
            "converged": final_grad_norm < precision,
            "alternating_detected": alternating_detected,
            "alternating_info": alternating_info,
            "final_grad_norm": final_grad_norm
        }

class MultiStartWeightedShapiroOptimizer:
    def __init__(self, manifold: SphereManifold, p: float):
        self.manifold = manifold
        self.p = p
        self.n = manifold.n
        self.rbfgs = WeightedShapiroRBFGS(manifold, p)
        
    def optimize_with_restarts(self, max_restarts: int = 10, 
                               perturbation_factor: float = 0.05,
                               max_iter_per_run: int = 1000) -> Dict:
        all_results = []
        
        y0 = np.random.randn(self.n)
        y0 = y0 / safe_norm(y0)
        
        print(f"\nRun 1/{max_restarts} (Initial random start)")
        result = self.rbfgs.optimize(
            y0=y0, 
            max_iter=max_iter_per_run,
            restart_on_alternating=True
        )
        all_results.append(result)
        current_best_obj = result["obj_opt"]
        current_best_y = result["y_opt"].copy()
        
        for restart_num in range(1, max_restarts):
            last_result = all_results[-1]
            
            if (last_result["alternating_detected"] and 
                last_result["obj_opt"] >= self.n - 1e-6):
                
                print(f"\nRun {restart_num+1}/{max_restarts} ({perturbation_factor*100:.0f}% perturbation on alternating pattern)")
                
                y_perturbed = self.manifold.perturb_alternating_solution(
                    last_result["y_opt"], 
                    perturbation_factor
                )
                
                result = self.rbfgs.optimize(
                    y0=y_perturbed, 
                    max_iter=max_iter_per_run,
                    restart_on_alternating=True
                )
            else:
                print(f"\nRun {restart_num+1}/{max_restarts} (Random restart)")
                y_random = np.random.randn(self.n)
                y_random = y_random / safe_norm(y_random)
                
                result = self.rbfgs.optimize(
                    y0=y_random, 
                    max_iter=max_iter_per_run,
                    restart_on_alternating=True
                )
            
            all_results.append(result)
            
            if result["obj_opt"] < current_best_obj:
                current_best_obj = result["obj_opt"]
                current_best_y = result["y_opt"].copy()
                print(f"  New best value found: {current_best_obj:.8f}")
            
            if result["obj_opt"] < self.n - 1e-6:
                print(f"  Solution violating inequality found! Objective = {result['obj_opt']:.8f} < n = {self.n}")
                #break
        
        total_iterations = sum(r["iter_num"] for r in all_results)
        total_time = sum(r["compute_time"] for r in all_results)
        
        alternating_runs = sum(1 for r in all_results if r["alternating_detected"])
        alternating_with_inequality = sum(
            1 for r in all_results 
            if r["alternating_detected"] and r["obj_opt"] >= self.n - 1e-6
        )
        
        inequality_satisfied = current_best_obj >= self.n - 1e-6
        
        return {
            "all_results": all_results,
            "best_y": current_best_y,
            "best_x": current_best_y ** 2,
            "best_obj": current_best_obj,
            "total_iterations": total_iterations,
            "total_time": total_time,
            "num_runs": len(all_results),
            "alternating_runs": alternating_runs,
            "alternating_with_inequality": alternating_with_inequality,
            "inequality_satisfied": inequality_satisfied,
            "n": self.n,
            "p": self.p,
            "q": 1 - self.p
        }

def save_results_to_excel(final_result: Dict, filename: str = None):
    output_dir = "shapiro_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = final_result["n"]
        p = final_result["p"]
        filename = f"shapiro_n{n}_p{p:.4f}_{timestamp}.xlsx"
    
    filepath = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        summary_data = {
            "Parameter": [
                "Dimension n", "Weight p", "Weight q", "Run Count", "Total Iterations", 
                "Total Time (s)", "Best Objective", "Trivial Case Bound n", "Difference",
                "Inequality Satisfied", "Alternating Pattern Runs", "Alternating Runs with Inequality"
            ],
            "Value": [
                final_result["n"], final_result["p"], final_result["q"], 
                final_result["num_runs"], final_result["total_iterations"],
                f"{final_result['total_time']:.6f}", f"{final_result['best_obj']:.10f}",
                final_result["n"], f"{final_result['best_obj'] - final_result['n']:.10f}",
                "Yes" if final_result["inequality_satisfied"] else "No",
                final_result["alternating_runs"], final_result["alternating_with_inequality"]
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        run_details_data = []
        for i, result in enumerate(final_result["all_results"]):
            run_details = {
                "Run Index": i+1,
                "Iterations": result["iter_num"],
                "Time (s)": f"{result['compute_time']:.6f}",
                "Final Objective": f"{result['obj_opt']:.10f}",
                "Final Gradient Norm": f"{result['final_grad_norm']:.6e}",
                "Converged": "Yes" if result["converged"] else "No",
                "Alternating Pattern Detected": "Yes" if result["alternating_detected"] else "No"
            }
            
            if result["alternating_detected"] and result["alternating_info"]:
                run_details["Alternating a Value"] = f"{result['alternating_info']['a']:.6f}"
                run_details["Alternating b Value"] = f"{result['alternating_info']['b']:.6f}"
            else:
                run_details["Alternating a Value"] = "-"
                run_details["Alternating b Value"] = "-"
            
            run_details_data.append(run_details)
        
        run_details_df = pd.DataFrame(run_details_data)
        run_details_df.to_excel(writer, sheet_name='Run Details', index=False)
        
        best_y = final_result["best_y"]
        best_x = final_result["best_x"]
        n = final_result["n"]
        p = final_result["p"]
        q = final_result["q"]
        
        ratios = []
        for i in range(n):
            i_next1 = (i + 1) % n
            i_next2 = (i + 2) % n
            numerator = best_x[i]
            denominator = p * best_x[i_next1] + q * best_x[i_next2]
            denominator = max(denominator, 1e-10)
            ratios.append(numerator / denominator)
        
        solution_data = []
        for i in range(n):
            solution_data.append({
                "Index i": i+1,
                "y_i": f"{best_y[i]:.10f}",
                "x_i = y_i²": f"{best_x[i]:.10f}",
                "Ratio x_i/(p·x_{i+1}+q·x_{i+2})": f"{ratios[i]:.10f}"
            })
        
        solution_df = pd.DataFrame(solution_data)
        solution_df.to_excel(writer, sheet_name='Optimal Solution', index=False)
        
        stats_data = {
            "Statistic": [
                "Min of y", "Max of y", "Mean of y", "Std of y",
                "Min of x", "Max of x", "Mean of x", "Std of x",
                "Min of Ratio", "Max of Ratio", "Mean of Ratio", "Std of Ratio",
                "Objective Value P(n,p,q)", "Trivial Case Bound n"
            ],
            "Value": [
                f"{np.min(best_y):.10f}", f"{np.max(best_y):.10f}", 
                f"{np.mean(best_y):.10f}", f"{np.std(best_y):.10f}",
                f"{np.min(best_x):.10f}", f"{np.max(best_x):.10f}", 
                f"{np.mean(best_x):.10f}", f"{np.std(best_x):.10f}",
                f"{np.min(ratios):.10f}", f"{np.max(ratios):.10f}", 
                f"{np.mean(ratios):.10f}", f"{np.std(ratios):.10f}",
                f"{final_result['best_obj']:.10f}", f"{final_result['n']:.10f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        if n <= 300:
            history_data = []
            for run_idx, result in enumerate(final_result["all_results"]):
                obj_history = result["obj_history"]
                grad_history = result["grad_norm_history"]
                
                num_points = min(200, len(obj_history))
                for i in range(num_points):
                    history_data.append({
                        "Run Index": run_idx + 1,
                        "Iteration": i + 1,
                        "Objective Value": f"{obj_history[i]:.10f}" if i < len(obj_history) else "",
                        "Gradient Norm": f"{grad_history[i]:.10e}" if i < len(grad_history) else ""
                    })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='Convergence History', index=False)
    
    print(f"\n✓ Results saved to: {filepath}")
    return filepath

def test_weighted_shapiro_with_restarts(n: int, p: float, max_restarts: int = 10, save_to_excel: bool = True):
    print("=" * 80)
    print(f"Denominator-Weighted Shapiro Cyclic Sum Optimization (n={n}, p={p:.4f}, q={1-p:.4f})")
    if n % 2 == 0:
        print("Note: n is even, alternating pattern detection and restart enabled")
    print("=" * 80)
    
    np.random.seed(42)
    manifold = SphereManifold(n=n)
    
    if n % 2 == 0:
        print(f"Using multi-start optimizer (max {max_restarts} restarts)...")
        optimizer = MultiStartWeightedShapiroOptimizer(manifold, p)
        final_result = optimizer.optimize_with_restarts(
            max_restarts=max_restarts,
            perturbation_factor=0.05
        )
    else:
        print("Using single optimization (n is odd)...")
        rbfgs = WeightedShapiroRBFGS(manifold=manifold, p=p)
        single_result = rbfgs.optimize(max_iter=1000, tol=1e-6)
        
        final_result = {
            "all_results": [single_result],
            "best_y": single_result["y_opt"],
            "best_x": single_result["x_opt"],
            "best_obj": single_result["obj_opt"],
            "total_iterations": single_result["iter_num"],
            "total_time": single_result["compute_time"],
            "num_runs": 1,
            "alternating_runs": 1 if single_result["alternating_detected"] else 0,
            "alternating_with_inequality": 1 if (
                single_result["alternating_detected"] and 
                single_result["obj_opt"] >= n - 1e-6
            ) else 0,
            "inequality_satisfied": single_result["obj_opt"] >= n - 1e-6,
            "n": n,
            "p": p,
            "q": 1 - p
        }
    
    print("\n" + "=" * 80)
    print("Optimization Results Summary")
    print("=" * 80)
    print(f"Number of runs: {final_result['num_runs']}")
    print(f"Total iterations: {final_result['total_iterations']}")
    print(f"Total computation time: {final_result['total_time']:.4f}s")
    print(f"Minimum P(n,p): {final_result['best_obj']:.10f}")
    print(f"Trivial case bound n: {n:.10f}")
    print(f"Difference between objective and trivial case bound: {final_result['best_obj'] - n:.10f}")
    
    if final_result['inequality_satisfied']:
        print(f"Inequality status: ✓ Satisfied P(n,p) >= n")
    else:
        print(f"Inequality status: ✗ Not satisfied P(n,p) >= n (Counterexample found!)")
    
    if n % 2 == 0:
        print(f"Runs with alternating pattern detected: {final_result['alternating_runs']}")
        print(f"Runs with alternating pattern and satisfied inequality: {final_result['alternating_with_inequality']}")
    
    print("\nRun details:")
    print("-" * 80)
    for i, result in enumerate(final_result['all_results']):
        print(f"Run {i+1}:")
        print(f"  Iterations: {result['iter_num']}")
        print(f"  Time: {result['compute_time']:.4f}s")
        print(f"  Final objective: {result['obj_opt']:.8f}")
        print(f"  Final gradient norm: {result['final_grad_norm']:.6e}")
        print(f"  Converged: {result['converged']}")
        if result['alternating_detected']:
            info = result['alternating_info']
            print(f"  Alternating pattern detected: a={info['a']:.4f}, b={info['b']:.4f}")
        print()
    
    print("=" * 80)
    
    if save_to_excel:
        excel_file = save_results_to_excel(final_result)
    
    plot_all_convergence_curves(final_result, n, p)
    plot_solution_characteristics(final_result, n, p)
    
    return final_result

def plot_all_convergence_curves(final_result: Dict, n: int, p: float):
    all_results = final_result['all_results']
    num_runs = len(all_results)
    
    cmap = get_cmap('tab10')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, result in enumerate(all_results):
        color = cmap(i % 10)
        grad_norm_history = result['grad_norm_history']
        
        ax1.plot(grad_norm_history, color=color, alpha=0.7, linewidth=1.5,
                label=f'Run {i+1}' if i < 10 else None)
        
        final_idx = len(grad_norm_history) - 1
        final_val = grad_norm_history[final_idx]
        ax1.scatter(final_idx, final_val, color=color, s=50, zorder=5)
    
    ax1.set_xlabel('Iterations (Cumulative)', fontsize=12)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title(f'Gradient Norm Convergence (n={n}, p={p:.4f}, {num_runs} runs)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    if num_runs <= 10:
        ax1.legend(fontsize=10, loc='upper right')
    
    cumulative_iter = 0
    for i, result in enumerate(all_results):
        color = cmap(i % 10)
        obj_history = result['obj_history']
        iterations = len(obj_history)
        
        x_coords = range(cumulative_iter, cumulative_iter + iterations)
        
        ax2.plot(x_coords, obj_history, color=color, alpha=0.7, linewidth=1.5,
                label=f'Run {i+1}' if i < 10 else None)
        
        final_val = obj_history[-1]
        ax2.scatter(cumulative_iter + iterations - 1, final_val, 
                   color=color, s=50, zorder=5)
        
        cumulative_iter += iterations
    
    ax2.axhline(y=n, color='r', linestyle='--', linewidth=2, 
                label=f'Trivial Case Bound n={n}', alpha=0.8)
    
    best_obj = final_result['best_obj']
    ax2.axhline(y=best_obj, color='g', linestyle=':', linewidth=2,
               label=f'Best Value {best_obj:.6f}', alpha=0.8)
    
    ax2.set_xlabel('Cumulative Iterations', fontsize=12)
    ax2.set_ylabel('P(n,p)', fontsize=12)
    ax2.set_title(f'Objective Convergence (n={n}, p={p:.4f}, {num_runs} runs)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if num_runs <= 10:
        ax2.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_solution_characteristics(final_result: Dict, n: int, p: float):
    best_y = final_result['best_y']
    best_x = best_y ** 2
    best_obj = final_result['best_obj']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    indices = np.arange(1, n+1)
    colors = ['red' if i % 2 == 0 else 'blue' for i in range(n)]
    ax1.bar(indices, best_y, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Index i', fontsize=12)
    ax1.set_ylabel('y_i', fontsize=12)
    ax1.set_title(f'Optimal Solution y Distribution (n={n})', fontsize=14)
    ax1.set_xticks(indices[::max(1, n//10)])
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[0, 1]
    ax2.bar(indices, best_x, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Index i', fontsize=12)
    ax2.set_ylabel('x_i = y_i²', fontsize=12)
    ax2.set_title(f'Optimal Solution x Distribution (Objective={best_obj:.6f})', fontsize=14)
    ax2.set_xticks(indices[::max(1, n//10)])
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    q = 1 - p
    ratios = []
    for i in range(n):
        i_next1 = (i + 1) % n
        i_next2 = (i + 2) % n
        numerator = best_x[i]
        denominator = p * best_x[i_next1] + q * best_x[i_next2]
        ratios.append(numerator / max(denominator, 1e-10))
    
    ax3.plot(indices, ratios, 'go-', linewidth=2, markersize=4)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Trivial Case Bound')
    ax3.axhline(y=np.mean(ratios), color='b', linestyle=':', alpha=0.5,
               label=f'Mean={np.mean(ratios):.3f}')
    
    ax3.set_xlabel('Index i', fontsize=12)
    ax3.set_ylabel('Ratio x_i/(p·x_{i+1}+q·x_{i+2})', fontsize=12)
    ax3.set_title('Term Ratio Distribution', fontsize=14)
    ax3.set_xticks(indices[::max(1, n//10)])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"Optimization Results:\n"
    stats_text += f"n = {n}\n"
    stats_text += f"p = {p:.4f}, q = {1-p:.4f}\n"
    stats_text += f"Runs: {final_result['num_runs']}\n"
    stats_text += f"Total Iterations: {final_result['total_iterations']}\n"
    stats_text += f"Total Time: {final_result['total_time']:.2f}s\n"
    stats_text += f"Best Objective: {best_obj:.10f}\n"
    stats_text += f"Trivial Case Bound: {n:.10f}\n"
    stats_text += f"Difference: {best_obj - n:.10f}\n\n"
    
    if final_result['inequality_satisfied']:
        stats_text += "Inequality: Satisfied\n"
    else:
        stats_text += "Inequality: Not Satisfied (Counterexample found!)\n"
        
    ax4.text(0.1, 0.95, stats_text, fontsize=17, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    #plt.suptitle(f'Denominator-Weighted Shapiro Sum Optimal Solution Analysis (n={n}, p={p:.4f})', fontsize=16)
    plt.tight_layout()
    plt.show()

def batch_test_shapiro(test_cases: List[Tuple[int, float]], max_restarts: int = 10):
    print("Starting Batch Test for Shapiro Cyclic Inequality")
    print("=" * 70)
    
    output_dir = "shapiro_batch_results"
    os.makedirs(output_dir, exist_ok=True)
    
    batch_summary = []
    all_results = []
    
    for idx, (n, p) in enumerate(test_cases):
        print(f"\nTest {idx+1}/{len(test_cases)}: n={n}, p={p:.4f}")
        print("-" * 50)
        
        try:
            result = test_weighted_shapiro_with_restarts(
                n=n, p=p, max_restarts=max_restarts, save_to_excel=False
            )
            
            batch_summary.append({
                "Index": idx+1,
                "n": n,
                "p": p,
                "q": 1-p,
                "Runs": result["num_runs"],
                "Total Iterations": result["total_iterations"],
                "Total Time (s)": result["total_time"],
                "Best Objective": result["best_obj"],
                "Trivial Case Bound": n,
                "Difference": result["best_obj"] - n,
                "Inequality Satisfied": "Yes" if result["inequality_satisfied"] else "No",
                "Alternating Pattern Runs": result["alternating_runs"] if n % 2 == 0 else "N/A",
                "Pattern with Inequality": result["alternating_with_inequality"] if n % 2 == 0 else "N/A"
            })
            
            filename = f"shapiro_n{n}_p{p:.4f}_batch.xlsx"
            save_results_to_excel(result, filename)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Test failed: {e}")
            batch_summary.append({
                "Index": idx+1,
                "n": n,
                "p": p,
                "q": 1-p,
                "Runs": "Failed",
                "Total Iterations": "Failed",
                "Total Time (s)": "Failed",
                "Best Objective": "Failed",
                "Trivial Case Bound": n,
                "Difference": "Failed",
                "Inequality Satisfied": "Failed",
                "Alternating Pattern Runs": "Failed",
                "Pattern with Inequality": "Failed"
            })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_summary_file = os.path.join(output_dir, f"shapiro_batch_summary_{timestamp}.xlsx")
    
    with pd.ExcelWriter(batch_summary_file, engine='openpyxl') as writer:
        summary_df = pd.DataFrame(batch_summary)
        summary_df.to_excel(writer, sheet_name='Batch Summary', index=False)
        
        detailed_results = []
        for i, result in enumerate(all_results):
            detailed_results.append({
                "Index": i+1,
                "n": result["n"],
                "p": result["p"],
                "Best Objective": result["best_obj"],
                "Inequality Satisfied": "Yes" if result["inequality_satisfied"] else "No",
                "Notes": "Test Successful"
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
    
    print(f"\n✓ Batch summary saved to: {batch_summary_file}")
    
    print("\n" + "=" * 70)
    print("Batch Test Summary")
    print("=" * 70)
    print(f"{'Idx':<4} {'n':<4} {'p':<8} {'Best Val':<12} {'Inequality':<10} {'Diff':<12}")
    print("-" * 70)
    
    for summary in batch_summary:
        if summary["Best Objective"] != "Failed":
            print(f"{summary['Index']:<4} {summary['n']:<4} {summary['p']:<8.4f} "
                  f"{summary['Best Objective']:<12.6f} {summary['Inequality Satisfied']:<10} "
                  f"{summary['Difference']:<12.6f}")
        else:
            print(f"{summary['Index']:<4} {summary['n']:<4} {summary['p']:<8.4f} "
                  f"{'Failed':<12} {'Failed':<10} {'Failed':<12}")

if __name__ == "__main__":
    print("Denominator-Weighted Shapiro Cyclic Sum Minimization")
    print("=" * 70)
    
    while True:
        print("\nSelect Mode:")
        print("1. Single Test")
        print("2. Batch Test Multiple n and p")
        print("3. Exit")
        
        choice = input("Enter selection (1/2/3): ").strip()
        
        if choice == '1':
            try:
                n = int(input("Enter dimension n (positive integer, n≥3): "))
                if n < 3:
                    print("n must be at least 3")
                    continue
                
                p = float(input("Enter weight p (0 < p < 1, e.g. 0.5): "))
                if p <= 0 or p >= 1:
                    print("p must be in the interval (0,1)")
                    continue
                
                max_restarts = 10
                if n % 2 == 0:
                    while True:
                        try:
                            user_input = input(f"Enter max restarts [Default={max_restarts}]: ").strip()
                            if user_input == "":
                                break
                            max_restarts = int(user_input)
                            if max_restarts < 1:
                                print("Restarts must be at least 1")
                            else:
                                break
                        except ValueError:
                            print("Please enter a valid integer")
                
                result = test_weighted_shapiro_with_restarts(n=n, p=p, max_restarts=max_restarts)
                
            except ValueError:
                print("Invalid input!")
        
        elif choice == '2':
            print("\nBatch Test Mode")
            print("Enter multiple test cases in format: n1,p1;n2,p2;...")
            print("Example: 3,0.5;4,0.5;5,0.5;6,0.5")
            
            while True:
                try:
                    input_str = input("Enter test cases: ").strip()
                    if input_str == "":
                        print("Input cannot be empty")
                        continue
                    
                    test_cases = []
                    parts = input_str.split(';')
                    for part in parts:
                        n_str, p_str = part.split(',')
                        n = int(n_str.strip())
                        p = float(p_str.strip())
                        
                        if n < 3:
                            print(f"n must be at least 3: n={n}")
                            continue
                        
                        if p <= 0 or p >= 1:
                            print(f"p must be in the interval (0,1): p={p}")
                            continue
                        
                        test_cases.append((n, p))
                    
                    if test_cases:
                        max_restarts = 10
                        restart_input = input(f"Enter max restarts [Default={max_restarts}]: ").strip()
                        if restart_input != "":
                            max_restarts = int(restart_input)
                        
                        batch_test_shapiro(test_cases, max_restarts)
                        break
                    else:
                        print("No valid test cases")
                        
                except ValueError as e:
                    print(f"Input format error: {e}")
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == '3':
            print("Exiting program.")
            break
        
        else:
            print("Invalid selection!")