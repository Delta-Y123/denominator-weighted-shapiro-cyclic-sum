import numpy as np
import time
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from datetime import datetime
import os

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 辅助函数：数值稳定性工具 --------------------------
def safe_norm(x: np.ndarray, eps: float = 1e-10) -> float:
    """安全计算范数，避免零范数"""
    norm_val = np.linalg.norm(x)
    return norm_val if norm_val > eps else eps

def safe_pow(x: np.ndarray, power: float, eps: float = 1e-10, min_val: float = 1e-8) -> np.ndarray:
    """安全幂运算，添加最小值限制"""
    x = np.where(np.abs(x) < eps, min_val, x)
    return np.power(x, power)

# -------------------------- 单位球面流形定义 --------------------------
class SphereManifold:
    """
    单位球面流形 S^{n-1} = {y ∈ ℝⁿ | ||y||² = 1}
    """
    def __init__(self, n: int):
        self.n = n  # 维度
        if n < 2:
            raise ValueError("维度n必须≥2")
        self.is_even = n % 2 == 0

    def proj(self, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """投影到y点的切空间：P_y(v) = v - <y,v>y"""
        return v - np.dot(y, v) * y

    def retraction(self, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """球面上的retraction：归一化"""
        y_new = y + eta
        return y_new / safe_norm(y_new)

    def weighted_shapiro_objective(self, y: np.ndarray, p: float) -> float:
        """
        分母加权Shapiro循环和目标函数
        变换: x_i = y_i^2 (保证非负)
        P(n,p,q) = sum_{i=1}^{n} x_i / (p*x_{i+1} + q*x_{i+2})
        其中 q = 1-p, 循环索引 x_{n+1}=x_1, x_{n+2}=x_2
        """
        q = 1.0 - p
        x = y ** 2  # x_i = y_i^2
        n = self.n
        
        obj_sum = 0.0
        for i in range(n):
            i_next1 = (i + 1) % n  # i+1 (循环)
            i_next2 = (i + 2) % n  # i+2 (循环)
            
            numerator = x[i]
            denominator = p * x[i_next1] + q * x[i_next2]
            
            # 数值稳定性：防止分母过小
            denominator = max(denominator, 1e-10)
            
            obj_sum += numerator / denominator
        
        return obj_sum

    def gradient(self, y: np.ndarray, p: float) -> np.ndarray:
        """
        计算欧氏梯度（对y求导），然后投影到切空间得到黎曼梯度
        
        设 f(y) = P(n,p,1-p) where x_i = y_i^2
        则 ∂f/∂y_i 需要用链式法则
        
        对于 x_i = y_i^2，有 dx_i/dy_i = 2*y_i
        """
        q = 1.0 - p
        x = y ** 2
        n = self.n
        
        # 欧氏梯度
        egrad = np.zeros(n)
        
        for i in range(n):
            # 第i项：x_i / (p*x_{i+1} + q*x_{i+2})
            # 对y_i的贡献：作为分子
            i_next1 = (i + 1) % n
            i_next2 = (i + 2) % n
            denom_i = p * x[i_next1] + q * x[i_next2]
            denom_i = max(denom_i, 1e-10)
            
            # ∂(x_i/denom_i)/∂y_i = (2*y_i) / denom_i
            egrad[i] += 2 * y[i] / denom_i
            
            # 对y_i的贡献：作为分母中的x_{i+1}
            # 第(i-1)项: x_{i-1} / (p*x_i + q*x_{i+1})
            i_prev1 = (i - 1) % n
            i_next1_from_prev = i  # 从i-1看，i+1就是i
            i_next2_from_prev = (i + 1) % n
            numer_prev1 = x[i_prev1]
            denom_prev1 = p * x[i] + q * x[i_next2_from_prev]
            denom_prev1 = max(denom_prev1, 1e-10)
            
            # ∂(x_{i-1}/denom)/∂y_i = -x_{i-1} * p * (2*y_i) / denom^2
            egrad[i] -= numer_prev1 * p * 2 * y[i] / (denom_prev1 ** 2)
            
            # 对y_i的贡献：作为分母中的x_{i+2}
            # 第(i-2)项: x_{i-2} / (p*x_{i-1} + q*x_i)
            i_prev2 = (i - 2) % n
            i_prev1_from_prev2 = (i - 1) % n
            numer_prev2 = x[i_prev2]
            denom_prev2 = p * x[i_prev1_from_prev2] + q * x[i]
            denom_prev2 = max(denom_prev2, 1e-10)
            
            # ∂(x_{i-2}/denom)/∂y_i = -x_{i-2} * q * (2*y_i) / denom^2
            egrad[i] -= numer_prev2 * q * 2 * y[i] / (denom_prev2 ** 2)
        
        # 投影到切空间得到黎曼梯度
        grad = self.proj(y, egrad)
        
        # 梯度裁剪防止爆炸
        grad = np.clip(grad, -1e3, 1e3)
        
        return grad

    def parallel_transport(self, y: np.ndarray, eta: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """球面上的平行传输"""
        y_new = self.retraction(y, eta)
        y_dot_ynew = np.dot(y, y_new)
        denominator = 1 + y_dot_ynew
        
        if abs(denominator) < 1e-10:
            return self.proj(y_new, xi)
        
        transport_term = np.dot(y + y_new, xi) / denominator * (y + y_new)
        return xi - transport_term
    
    def is_alternating_pattern(self, y: np.ndarray, tol: float = 0.01) -> Tuple[bool, float, float]:
        """
        检测解是否为(a,b,a,b,...)型交替模式
        
        参数:
            y: 待检测的解向量
            tol: 允许的相对差异容差
        
        返回:
            (is_alternating, a_value, b_value): 
                is_alternating: 是否为交替模式
                a_value: 奇数位置的平均值
                b_value: 偶数位置的平均值
        """
        if not self.is_even:
            return False, 0.0, 0.0
        
        n = self.n
        # 计算奇数位置和偶数位置的值
        odd_values = y[0:n:2]  # 索引0,2,4,...
        even_values = y[1:n:2]  # 索引1,3,5,...
        
        # 计算各自的标准差
        odd_std = np.std(odd_values)
        even_std = np.std(even_values)
        
        # 计算各自的值范围
        odd_range = np.max(odd_values) - np.min(odd_values)
        even_range = np.max(even_values) - np.min(even_values)
        
        # 检查是否满足交替模式条件
        odd_mean = np.mean(odd_values)
        even_mean = np.mean(even_values)
        
        # 如果奇数和偶数各自内部的差异很小，且奇数和偶数之间的差异相对较大
        odd_uniform = odd_std / (abs(odd_mean) + 1e-10) < tol
        even_uniform = even_std / (abs(even_mean) + 1e-10) < tol
        
        # 计算奇偶之间的相对差异
        if abs(odd_mean) > 1e-10 and abs(even_mean) > 1e-10:
            relative_diff = abs(odd_mean - even_mean) / max(abs(odd_mean), abs(even_mean))
        else:
            relative_diff = abs(odd_mean - even_mean)
        
        is_alternating = odd_uniform and even_uniform and relative_diff > 0.1
        
        return is_alternating, odd_mean, even_mean
    
    def perturb_alternating_solution(self, y: np.ndarray, perturbation_factor: float = 0.05) -> np.ndarray:
        """
        对交替模式解进行摄动
        
        参数:
            y: 原始解向量
            perturbation_factor: 摄动因子 (例如0.05表示5%摄动)
        
        返回:
            摄动后的解向量 (已归一化到单位球面)
        """
        n = len(y)
        # 添加随机摄动
        perturbation = np.random.randn(n) * perturbation_factor
        y_perturbed = y + perturbation
        
        # 重新归一化到单位球面
        y_perturbed = y_perturbed / safe_norm(y_perturbed)
        
        return y_perturbed

# -------------------------- 黎曼BFGS优化器 --------------------------
class WeightedShapiroRBFGS:
    """针对分母加权Shapiro问题的RBFGS优化器"""
    def __init__(self, manifold: SphereManifold, p: float):
        self.manifold = manifold
        self.p = p
        self.n = manifold.n

    def _objective(self, y: np.ndarray) -> float:
        """目标函数"""
        return self.manifold.weighted_shapiro_objective(y, self.p)

    def _line_search_armijo(self, y: np.ndarray, eta: np.ndarray, grad: np.ndarray) -> float:
        """Armijo线搜索"""
        c1 = 1e-4
        alpha = 1.0
        max_ls_iter = 20
        f_y = self._objective(y)
        grad_dot_eta = np.dot(grad, eta)

        # 确保搜索方向是下降方向
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
        """BFGS逆Hessian更新"""
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
        """
        优化主循环
        
        参数:
            restart_on_alternating: 是否在检测到交替模式且P(n,p,q)>=n时重新启动
        """
        # 初始化
        if y0 is None:
            y0 = np.random.randn(self.n)
            y0 = y0 / safe_norm(y0)
        else:
            y0 = y0 / safe_norm(y0)

        y = y0.copy()
        H = np.eye(self.n)
        
        grad_norm_history = []
        obj_history = []
        start_time = time.time()
        
        prev_grad = None
        prev_eta = None
        prev_y = None
        
        # 记录是否检测到交替模式
        alternating_detected = False
        alternating_info = None

        for k in range(max_iter):
            # 计算梯度
            grad = self.manifold.gradient(y, self.p)
            grad_norm = np.linalg.norm(grad)
            grad_norm_history.append(grad_norm)
            
            # 计算目标函数值
            obj_val = self._objective(y)
            obj_history.append(obj_val)
            
            # 检查是否为交替模式 (在偶数n且优化进行到后期时检查)
            if (self.n % 2 == 0 and restart_on_alternating and 
                k > 50 and grad_norm < 1e-2):  # 梯度较小且迭代次数较多时检查
                is_alternating, a_val, b_val = self.manifold.is_alternating_pattern(y)
                if is_alternating:
                    alternating_detected = True
                    alternating_info = {
                        'a': a_val,
                        'b': b_val,
                        'iteration': k,
                        'objective': obj_val
                    }
                    print(f"  检测到交替模式: a={a_val:.4f}, b={b_val:.4f}, 目标值={obj_val:.6f}")
                    
                    # 如果满足不等式且是交替模式，提前终止本次优化
                    if obj_val >= self.n - 1e-6:
                        print(f"  交替模式满足不等式，提前终止本次优化")
                        break

            # 收敛检查
            if grad_norm < tol:
                break

            # 计算搜索方向
            if k == 0:
                eta = -grad
            else:
                eta = -H @ grad

            # 线搜索
            alpha = self._line_search_armijo(y, eta, grad)

            # 更新点
            y_new = self.manifold.retraction(y, alpha * eta)

            # BFGS更新
            if k > 0 and k < max_bfgs_iter:
                # 计算s和y
                s_transported = self.manifold.parallel_transport(prev_y, alpha * prev_eta, alpha * prev_eta)
                grad_transported = self.manifold.parallel_transport(prev_y, alpha * prev_eta, prev_grad)
                grad_new = self.manifold.gradient(y_new, self.p)
                y_vec = grad_new - grad_transported

                # BFGS更新
                H = self._bfgs_update(H, s_transported, y_vec)

            # 保存状态
            prev_grad = grad.copy()
            prev_eta = eta.copy()
            prev_y = y.copy()
            y = y_new

        # 返回结果
        end_time = time.time()
        final_grad = self.manifold.gradient(y, self.p)
        final_grad_norm = np.linalg.norm(final_grad)
        final_obj = self._objective(y)

        return {
            "y_opt": y,
            "x_opt": y ** 2,  # 最优的x值
            "obj_opt": final_obj,
            "iter_num": k + 1,
            "compute_time": end_time - start_time,
            "grad_norm_history": np.array(grad_norm_history),
            "obj_history": np.array(obj_history),
            "converged": final_grad_norm < tol,
            "alternating_detected": alternating_detected,
            "alternating_info": alternating_info,
            "final_grad_norm": final_grad_norm
        }

# -------------------------- 多起点优化器 --------------------------
class MultiStartWeightedShapiroOptimizer:
    """多起点优化器，专门处理交替模式问题"""
    def __init__(self, manifold: SphereManifold, p: float):
        self.manifold = manifold
        self.p = p
        self.n = manifold.n
        self.rbfgs = WeightedShapiroRBFGS(manifold, p)
        
    def optimize_with_restarts(self, max_restarts: int = 10, 
                               perturbation_factor: float = 0.05,
                               max_iter_per_run: int = 1000) -> Dict:
        """
        进行多次优化，每次检测到交替模式且满足不等式时进行摄动重启
        
        返回:
            包含所有运行结果和最终最优解的字典
        """
        # 存储所有运行结果
        all_results = []
        
        # 初始随机起点
        y0 = np.random.randn(self.n)
        y0 = y0 / safe_norm(y0)
        
        # 第一次运行
        print(f"\n运行 1/{max_restarts} (初始随机起点)")
        result = self.rbfgs.optimize(
            y0=y0, 
            max_iter=max_iter_per_run,
            restart_on_alternating=True
        )
        all_results.append(result)
        current_best_obj = result["obj_opt"]
        current_best_y = result["y_opt"].copy()
        
        # 后续运行（如果需要）
        for restart_num in range(1, max_restarts):
            # 检查上次运行是否检测到交替模式且满足不等式
            last_result = all_results[-1]
            
            if (last_result["alternating_detected"] and 
                last_result["obj_opt"] >= self.n - 1e-6):
                
                print(f"\n运行 {restart_num+1}/{max_restarts} (对交替模式进行{perturbation_factor*100:.0f}%摄动)")
                
                # 对交替模式解进行摄动
                y_perturbed = self.manifold.perturb_alternating_solution(
                    last_result["y_opt"], 
                    perturbation_factor
                )
                
                # 使用摄动后的点作为新起点
                result = self.rbfgs.optimize(
                    y0=y_perturbed, 
                    max_iter=max_iter_per_run,
                    restart_on_alternating=True
                )
            else:
                # 如果没有检测到交替模式或不等式不成立，则随机重启
                print(f"\n运行 {restart_num+1}/{max_restarts} (随机重启)")
                y_random = np.random.randn(self.n)
                y_random = y_random / safe_norm(y_random)
                
                result = self.rbfgs.optimize(
                    y0=y_random, 
                    max_iter=max_iter_per_run,
                    restart_on_alternating=True
                )
            
            all_results.append(result)
            
            # 更新最优解
            if result["obj_opt"] < current_best_obj:
                current_best_obj = result["obj_opt"]
                current_best_y = result["y_opt"].copy()
                print(f"  找到新的最优值: {current_best_obj:.8f}")
            
            # 如果找到违反不等式的解（即目标值 < n），可以提前终止
            if result["obj_opt"] < self.n - 1e-6:
                print(f"  找到违反不等式的解! 目标值 = {result['obj_opt']:.8f} < n = {self.n}")
                break
        
        # 汇总结果
        total_iterations = sum(r["iter_num"] for r in all_results)
        total_time = sum(r["compute_time"] for r in all_results)
        
        # 统计交替模式检测情况
        alternating_runs = sum(1 for r in all_results if r["alternating_detected"])
        alternating_with_inequality = sum(
            1 for r in all_results 
            if r["alternating_detected"] and r["obj_opt"] >= self.n - 1e-6
        )
        
        # 检查最终是否满足不等式
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

# -------------------------- Excel输出函数 --------------------------
def save_results_to_excel(final_result: Dict, filename: str = None):
    """
    将优化结果保存到Excel文件
    
    参数:
        final_result: 优化结果字典
        filename: Excel文件名，如果为None则自动生成
    """
    # 创建输出目录
    output_dir = "shapiro_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = final_result["n"]
        p = final_result["p"]
        filename = f"shapiro_n{n}_p{p:.4f}_{timestamp}.xlsx"
    
    filepath = os.path.join(output_dir, filename)
    
    # 创建Excel写入器
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 1. 创建汇总表 - 总体信息
        summary_data = {
            "参数": [
                "维数 n", "权重 p", "权重 q", "运行次数", "总迭代次数", 
                "总计算时间(秒)", "最优目标值", "理论下界 n", "差值",
                "不等式是否成立", "检测到交替模式的运行", "交替模式且满足不等式的运行"
            ],
            "值": [
                final_result["n"], final_result["p"], final_result["q"], 
                final_result["num_runs"], final_result["total_iterations"],
                f"{final_result['total_time']:.6f}", f"{final_result['best_obj']:.10f}",
                final_result["n"], f"{final_result['best_obj'] - final_result['n']:.10f}",
                "是" if final_result["inequality_satisfied"] else "否",
                final_result["alternating_runs"], final_result["alternating_with_inequality"]
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='汇总信息', index=False)
        
        # 2. 创建每次运行的详细信息表
        run_details_data = []
        for i, result in enumerate(final_result["all_results"]):
            run_details = {
                "运行序号": i+1,
                "迭代次数": result["iter_num"],
                "计算时间(秒)": f"{result['compute_time']:.6f}",
                "最终目标值": f"{result['obj_opt']:.10f}",
                "最终梯度范数": f"{result['final_grad_norm']:.6e}",
                "是否收敛": "是" if result["converged"] else "否",
                "检测到交替模式": "是" if result["alternating_detected"] else "否"
            }
            
            # 添加交替模式详细信息
            if result["alternating_detected"] and result["alternating_info"]:
                run_details["交替模式a值"] = f"{result['alternating_info']['a']:.6f}"
                run_details["交替模式b值"] = f"{result['alternating_info']['b']:.6f}"
            else:
                run_details["交替模式a值"] = "-"
                run_details["交替模式b值"] = "-"
            
            run_details_data.append(run_details)
        
        run_details_df = pd.DataFrame(run_details_data)
        run_details_df.to_excel(writer, sheet_name='各次运行详情', index=False)
        
        # 3. 创建最优解详细信息表
        best_y = final_result["best_y"]
        best_x = final_result["best_x"]
        n = final_result["n"]
        p = final_result["p"]
        q = final_result["q"]
        
        # 计算比值
        ratios = []
        for i in range(n):
            i_next1 = (i + 1) % n
            i_next2 = (i + 2) % n
            numerator = best_x[i]
            denominator = p * best_x[i_next1] + q * best_x[i_next2]
            denominator = max(denominator, 1e-10)
            ratios.append(numerator / denominator)
        
        # 创建最优解数据
        solution_data = []
        for i in range(n):
            solution_data.append({
                "索引 i": i+1,
                "y_i": f"{best_y[i]:.10f}",
                "x_i = y_i²": f"{best_x[i]:.10f}",
                "比值 x_i/(p·x_{i+1}+q·x_{i+2})": f"{ratios[i]:.10f}"
            })
        
        solution_df = pd.DataFrame(solution_data)
        solution_df.to_excel(writer, sheet_name='最优解详细信息', index=False)
        
        # 4. 创建统计信息表
        stats_data = {
            "统计量": [
                "y的最小值", "y的最大值", "y的平均值", "y的标准差",
                "x的最小值", "x的最大值", "x的平均值", "x的标准差",
                "比值的最小值", "比值的最大值", "比值的平均值", "比值的标准差",
                "目标函数值 P(n,p,q)", "理论下界 n"
            ],
            "值": [
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
        stats_df.to_excel(writer, sheet_name='统计信息', index=False)
        
        # 5. 如果n不是太大，创建历史数据表
        if n <= 300:  # 避免数据量过大
            # 收集每次运行的历史数据
            history_data = []
            for run_idx, result in enumerate(final_result["all_results"]):
                obj_history = result["obj_history"]
                grad_history = result["grad_norm_history"]
                
                # 取前200个点或全部（如果少于200个）
                num_points = min(200, len(obj_history))
                for i in range(num_points):
                    history_data.append({
                        "运行序号": run_idx + 1,
                        "迭代次数": i + 1,
                        "目标函数值": f"{obj_history[i]:.10f}" if i < len(obj_history) else "",
                        "梯度范数": f"{grad_history[i]:.10e}" if i < len(grad_history) else ""
                    })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='收敛历史', index=False)
    
    print(f"\n✓ 结果已保存到: {filepath}")
    return filepath

# -------------------------- 主测试函数 --------------------------
def test_weighted_shapiro_with_restarts(n: int, p: float, max_restarts: int = 10, save_to_excel: bool = True):
    """
    测试分母加权Shapiro循环不等式（带重启功能）
    
    参数:
        n: 维数
        p: 权重
        max_restarts: 最大重启次数
        save_to_excel: 是否保存结果到Excel
    """
    print("=" * 80)
    print(f"分母加权Shapiro循环不等式优化 (n={n}, p={p:.4f}, q={1-p:.4f})")
    if n % 2 == 0:
        print("注意: n为偶数，将启用交替模式检测和重启功能")
    print("=" * 80)
    
    # 初始化
    np.random.seed(42)
    manifold = SphereManifold(n=n)
    
    # 选择优化策略
    if n % 2 == 0:
        # 对于偶数n，使用多起点优化器
        print(f"使用多起点优化器 (最多{max_restarts}次重启)...")
        optimizer = MultiStartWeightedShapiroOptimizer(manifold, p)
        final_result = optimizer.optimize_with_restarts(
            max_restarts=max_restarts,
            perturbation_factor=0.05
        )
    else:
        # 对于奇数n，使用单次优化
        print("使用单次优化 (n为奇数)...")
        rbfgs = WeightedShapiroRBFGS(manifold=manifold, p=p)
        single_result = rbfgs.optimize(max_iter=1000, tol=1e-6)
        
        # 转换为与多起点相同的格式
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
    
    # 输出汇总结果
    print("\n" + "=" * 80)
    print("优化结果汇总")
    print("=" * 80)
    print(f"运行次数: {final_result['num_runs']}")
    print(f"总迭代次数: {final_result['total_iterations']}")
    print(f"总计算时间: {final_result['total_time']:.4f}秒")
    print(f"P(n,p,1-p)的最小值: {final_result['best_obj']:.10f}")
    print(f"理论下界 n/(p+q): {n:.10f}")
    print(f"目标值与下界的差: {final_result['best_obj'] - n:.10f}")
    
    if final_result['inequality_satisfied']:
        print(f"不等式状态: ✓ 满足 P(n,p,1-p) >= n")
    else:
        print(f"不等式状态: ✗ 不满足 P(n,p,1-p) >= n (找到反例!)")
    
    if n % 2 == 0:
        print(f"检测到交替模式的运行次数: {final_result['alternating_runs']}")
        print(f"交替模式且满足不等式的运行次数: {final_result['alternating_with_inequality']}")
    
    # 输出每次运行的详细信息
    print("\n各次运行详情:")
    print("-" * 80)
    for i, result in enumerate(final_result['all_results']):
        print(f"运行 {i+1}:")
        print(f"  迭代次数: {result['iter_num']}")
        print(f"  计算时间: {result['compute_time']:.4f}秒")
        print(f"  最终目标值: {result['obj_opt']:.8f}")
        print(f"  最终梯度范数: {result['final_grad_norm']:.6e}")
        print(f"  是否收敛: {result['converged']}")
        if result['alternating_detected']:
            info = result['alternating_info']
            print(f"  检测到交替模式: a={info['a']:.4f}, b={info['b']:.4f}")
        print()
    
    print("=" * 80)
    
    # 保存结果到Excel
    if save_to_excel:
        excel_file = save_results_to_excel(final_result)
    
    # 绘制所有运行的收敛曲线
    plot_all_convergence_curves(final_result, n, p)
    
    # 绘制最优解的特征
    plot_solution_characteristics(final_result, n, p)
    
    return final_result

def plot_all_convergence_curves(final_result: Dict, n: int, p: float):
    """绘制所有运行的收敛曲线"""
    all_results = final_result['all_results']
    num_runs = len(all_results)
    
    # 创建颜色映射
    cmap = get_cmap('tab10')
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：梯度范数收敛曲线
    for i, result in enumerate(all_results):
        color = cmap(i % 10)
        grad_norm_history = result['grad_norm_history']
        
        # 绘制梯度范数历史
        ax1.plot(grad_norm_history, color=color, alpha=0.7, linewidth=1.5,
                label=f'运行 {i+1}' if i < 10 else None)
        
        # 标记最终点
        final_idx = len(grad_norm_history) - 1
        final_val = grad_norm_history[final_idx]
        ax1.scatter(final_idx, final_val, color=color, s=50, zorder=5)
    
    ax1.set_xlabel('迭代次数 (累计)', fontsize=12)
    ax1.set_ylabel('梯度范数', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title(f'梯度范数收敛曲线 (n={n}, p={p:.4f}, {num_runs}次运行)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    if num_runs <= 10:
        ax1.legend(fontsize=10, loc='upper right')
    
    # 右图：目标函数收敛曲线
    cumulative_iter = 0
    for i, result in enumerate(all_results):
        color = cmap(i % 10)
        obj_history = result['obj_history']
        iterations = len(obj_history)
        
        # 创建x轴坐标（累计迭代次数）
        x_coords = range(cumulative_iter, cumulative_iter + iterations)
        
        # 绘制目标函数历史
        ax2.plot(x_coords, obj_history, color=color, alpha=0.7, linewidth=1.5,
                label=f'运行 {i+1}' if i < 10 else None)
        
        # 标记最终点
        final_val = obj_history[-1]
        ax2.scatter(cumulative_iter + iterations - 1, final_val, 
                   color=color, s=50, zorder=5)
        
        cumulative_iter += iterations
    
    # 添加理论下界线
    ax2.axhline(y=n, color='r', linestyle='--', linewidth=2, 
                label=f'理论下界 n={n}', alpha=0.8)
    
    # 标记最优解
    best_obj = final_result['best_obj']
    ax2.axhline(y=best_obj, color='g', linestyle=':', linewidth=2,
               label=f'最优值 {best_obj:.6f}', alpha=0.8)
    
    ax2.set_xlabel('累计迭代次数', fontsize=12)
    ax2.set_ylabel('P(n,p,1-p)', fontsize=12)
    ax2.set_title(f'目标函数收敛曲线 (n={n}, p={p:.4f}, {num_runs}次运行)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if num_runs <= 10:
        ax2.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_solution_characteristics(final_result: Dict, n: int, p: float):
    """绘制最优解的特征"""
    best_y = final_result['best_y']
    best_x = best_y ** 2
    best_obj = final_result['best_obj']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：y值分布
    ax1 = axes[0, 0]
    indices = np.arange(1, n+1)
    colors = ['red' if i % 2 == 0 else 'blue' for i in range(n)]
    ax1.bar(indices, best_y, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('索引 i', fontsize=12)
    ax1.set_ylabel('y_i', fontsize=12)
    ax1.set_title(f'最优解 y 值分布 (n={n})', fontsize=14)
    ax1.set_xticks(indices[::max(1, n//10)])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右上：x值分布
    ax2 = axes[0, 1]
    ax2.bar(indices, best_x, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('索引 i', fontsize=12)
    ax2.set_ylabel('x_i = y_i²', fontsize=12)
    ax2.set_title(f'最优解 x 值分布 (目标值={best_obj:.6f})', fontsize=14)
    ax2.set_xticks(indices[::max(1, n//10)])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 左下：比值分布
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
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理论下界比')
    ax3.axhline(y=np.mean(ratios), color='b', linestyle=':', alpha=0.5,
               label=f'均值={np.mean(ratios):.3f}')
    
    ax3.set_xlabel('索引 i', fontsize=12)
    ax3.set_ylabel('比值 x_i/(p·x_{i+1}+q·x_{i+2})', fontsize=12)
    ax3.set_title('各项比值分布', fontsize=14)
    ax3.set_xticks(indices[::max(1, n//10)])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 右下：运行统计
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 显示统计信息
    stats_text = f"优化结果统计:\n"
    stats_text += f"n = {n}\n"
    stats_text += f"p = {p:.4f}, q = {1-p:.4f}\n"
    stats_text += f"运行次数: {final_result['num_runs']}\n"
    stats_text += f"总迭代次数: {final_result['total_iterations']}\n"
    stats_text += f"总计算时间: {final_result['total_time']:.2f}秒\n"
    stats_text += f"最优目标值: {best_obj:.10f}\n"
    stats_text += f"理论下界: {n:.10f}\n"
    stats_text += f"差值: {best_obj - n:.10f}\n\n"
    
    if final_result['inequality_satisfied']:
        stats_text += "不等式状态: ✓ 成立\n"
    else:
        stats_text += "不等式状态: ✗ 不成立 (找到反例!)\n"
    
    if n % 2 == 0:
        stats_text += f"\n交替模式统计:\n"
        stats_text += f"检测到交替模式的运行: {final_result['alternating_runs']}\n"
        stats_text += f"交替模式且满足不等式: {final_result['alternating_with_inequality']}"
    
    ax4.text(0.1, 0.95, stats_text, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'加权Shapiro问题最优解特征分析 (n={n}, p={p:.4f})', fontsize=16)
    plt.tight_layout()
    plt.show()

# -------------------------- 批量测试函数 --------------------------
def batch_test_shapiro(test_cases: List[Tuple[int, float]], max_restarts: int = 10):
    """
    批量测试多个n和p的组合
    
    参数:
        test_cases: 测试用例列表，每个元素为(n, p)
        max_restarts: 最大重启次数
    """
    print("开始批量测试Shapiro循环不等式")
    print("=" * 70)
    
    # 创建结果目录
    output_dir = "shapiro_batch_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建汇总DataFrame
    batch_summary = []
    all_results = []
    
    for idx, (n, p) in enumerate(test_cases):
        print(f"\n测试 {idx+1}/{len(test_cases)}: n={n}, p={p:.4f}")
        print("-" * 50)
        
        try:
            # 运行测试
            result = test_weighted_shapiro_with_restarts(
                n=n, p=p, max_restarts=max_restarts, save_to_excel=False
            )
            
            # 添加到汇总
            batch_summary.append({
                "序号": idx+1,
                "n": n,
                "p": p,
                "q": 1-p,
                "运行次数": result["num_runs"],
                "总迭代次数": result["total_iterations"],
                "总计算时间(秒)": result["total_time"],
                "最优目标值": result["best_obj"],
                "理论下界": n,
                "差值": result["best_obj"] - n,
                "不等式是否成立": "是" if result["inequality_satisfied"] else "否",
                "检测到交替模式运行": result["alternating_runs"] if n % 2 == 0 else "N/A",
                "交替模式满足不等式": result["alternating_with_inequality"] if n % 2 == 0 else "N/A"
            })
            
            # 保存当前测试的Excel文件
            filename = f"shapiro_n{n}_p{p:.4f}_batch.xlsx"
            save_results_to_excel(result, filename)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"测试失败: {e}")
            batch_summary.append({
                "序号": idx+1,
                "n": n,
                "p": p,
                "q": 1-p,
                "运行次数": "失败",
                "总迭代次数": "失败",
                "总计算时间(秒)": "失败",
                "最优目标值": "失败",
                "理论下界": n,
                "差值": "失败",
                "不等式是否成立": "失败",
                "检测到交替模式运行": "失败",
                "交替模式满足不等式": "失败"
            })
    
    # 保存批量测试汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_summary_file = os.path.join(output_dir, f"shapiro_batch_summary_{timestamp}.xlsx")
    
    # 创建Excel写入器
    with pd.ExcelWriter(batch_summary_file, engine='openpyxl') as writer:
        # 汇总表
        summary_df = pd.DataFrame(batch_summary)
        summary_df.to_excel(writer, sheet_name='批量测试汇总', index=False)
        
        # 详细结果表（如果需要）
        detailed_results = []
        for i, result in enumerate(all_results):
            detailed_results.append({
                "序号": i+1,
                "n": result["n"],
                "p": result["p"],
                "最优目标值": result["best_obj"],
                "不等式是否成立": "是" if result["inequality_satisfied"] else "否",
                "备注": "测试成功"
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_excel(writer, sheet_name='详细结果', index=False)
    
    print(f"\n✓ 批量测试汇总结果已保存到: {batch_summary_file}")
    
    # 打印汇总结果
    print("\n" + "=" * 70)
    print("批量测试汇总结果")
    print("=" * 70)
    print(f"{'序号':<4} {'n':<4} {'p':<8} {'最优值':<12} {'不等式':<8} {'差值':<12}")
    print("-" * 70)
    
    for summary in batch_summary:
        if summary["最优目标值"] != "失败":
            print(f"{summary['序号']:<4} {summary['n']:<4} {summary['p']:<8.4f} "
                  f"{summary['最优目标值']:<12.6f} {summary['不等式是否成立']:<8} "
                  f"{summary['差值']:<12.6f}")
        else:
            print(f"{summary['序号']:<4} {summary['n']:<4} {summary['p']:<8.4f} "
                  f"{'失败':<12} {'失败':<8} {'失败':<12}")

# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    print("分母加权Shapiro循环不等式最小值计算程序 (带交替模式检测和重启)")
    print("=" * 70)
    print("说明: 当n为偶数时，程序会检测(a,b,a,b,...)型交替模式解。")
    print("      如果交替模式解满足不等式，程序会对其进行摄动并重新优化。")
    print("      结果将自动保存到Excel文件。")
    print("=" * 70)
    
    while True:
        print("\n选择模式:")
        print("1. 单次测试")
        print("2. 批量测试多个n和p")
        print("3. 退出")
        
        choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == '1':
            try:
                n = int(input("请输入维数n (正整数, n≥3): "))
                if n < 3:
                    print("n必须至少为3")
                    continue
                
                p = float(input("请输入权重p (0 < p < 1, 例如0.5): "))
                if p <= 0 or p >= 1:
                    print("p必须在(0,1)区间内")
                    continue
                
                # 设置最大重启次数（仅对偶数n有效）
                max_restarts = 10
                if n % 2 == 0:
                    while True:
                        try:
                            user_input = input(f"请输入最大重启次数 [默认={max_restarts}]: ").strip()
                            if user_input == "":
                                break
                            max_restarts = int(user_input)
                            if max_restarts < 1:
                                print("重启次数必须至少为1")
                            else:
                                break
                        except ValueError:
                            print("请输入有效的整数")
                
                # 运行优化
                result = test_weighted_shapiro_with_restarts(n=n, p=p, max_restarts=max_restarts)
                
            except ValueError:
                print("输入无效！")
        
        elif choice == '2':
            print("\n批量测试模式")
            print("输入多个测试用例，格式为: n1,p1;n2,p2;...")
            print("例如: 3,0.5;4,0.5;5,0.5;6,0.5")
            
            while True:
                try:
                    input_str = input("请输入测试用例: ").strip()
                    if input_str == "":
                        print("输入不能为空")
                        continue
                    
                    test_cases = []
                    parts = input_str.split(';')
                    for part in parts:
                        n_str, p_str = part.split(',')
                        n = int(n_str.strip())
                        p = float(p_str.strip())
                        
                        if n < 3:
                            print(f"n必须至少为3: n={n}")
                            continue
                        
                        if p <= 0 or p >= 1:
                            print(f"p必须在(0,1)区间内: p={p}")
                            continue
                        
                        test_cases.append((n, p))
                    
                    if test_cases:
                        max_restarts = 10
                        restart_input = input(f"请输入最大重启次数 [默认={max_restarts}]: ").strip()
                        if restart_input != "":
                            max_restarts = int(restart_input)
                        
                        batch_test_shapiro(test_cases, max_restarts)
                        break
                    else:
                        print("没有有效的测试用例")
                        
                except ValueError as e:
                    print(f"输入格式错误: {e}")
                except Exception as e:
                    print(f"错误: {e}")
        
        elif choice == '3':
            print("退出程序。")
            break
        
        else:
            print("无效选择！")