function plot_region(p)
    % 绘制区域 {(u,v)|1<v<=1+(p/q)u-(p^3 / q^3)u^2}，其中q=1-p
    %
    % 输入:
    %   p - 标量值，在区间 [0, 1] 内

    % 检查输入参数 p 的有效性
    if p < 0 || p > 1
        error('p 必须在区间 [0, 1] 内。');
    end

    % 计算 q
    q = 1 - p;

    % 处理边界情况
    if p == 1
        fprintf('当 p=1 时, q=0，(p/q) 和 (p^3/q^3) 未定义。\n');
        fprintf('该区域退化为一个空集。\n');
        
        % 创建一个图来展示这个特殊情况
        figure;
        title('p = 1; q = 0; 区域未定义（空集）');
        xlabel('u');
        ylabel('v');
        grid on;
        return; % 结束函数
    elseif p == 0
        fprintf('当 p=0 时, q=1。不等式变为 1 < v <= 1。\n');
        fprintf('该区域退化为一个空集。\n');
        
        % 创建一个图来展示这个特殊情况
        figure;
        title('p = 0; q = 1; 区域 v > 1 且 v <= 1 是空集');
        xlabel('u');
        ylabel('v');
        grid on;
        return; % 结束函数
    end

    % --- 对于 0 < p < 1 的一般情况 ---
    
    % 定义 u 的范围
    % 抛物线 1 + (p/q)u - (p^3 / q^3)u^2 在 u=0 和 u=q/p 时等于 1，
    % 所以我们选择一个覆盖这个区间的范围。
    u_min = -0.5 * (q / p);
    u_max = 1.8 * (q / p);
    
    % 创建网格点用于绘制
    num_points = 500;
    u_vec = linspace(u_min, u_max, num_points);
    v_vec = linspace(0.9, 1.3, num_points); % v 范围根据预期图形调整
    
    [U_mesh, V_mesh] = meshgrid(u_vec, v_vec);

    % 计算不等式的右半部分
    % 注意：为了数值稳定性，将公式重写为 1 + (p/q)*u*(1 - (p^2 / q^2)*u)
    parabola_values = 1 + (p/q) * U_mesh .* (1 - (p^2 / q^2) * U_mesh);

    % 定义区域的布尔掩码
    % 条件是 1 < v AND v <= parabola_value
    region_mask = (V_mesh > 1) & (V_mesh <= parabola_values);

    % 绘图
    figure;
    hold on;
    
    % 使用 surf 或 pcolor 绘制区域
    % 将非区域部分设置为 NaN，使其不显示
    v_to_plot = V_mesh;
    v_to_plot(~region_mask) = NaN;
    pcolor(U_mesh, V_mesh, v_to_plot);
    shading flat;
    colormap([0.7 0.7 1]); % 淡蓝色填充
    
    % 为了更清晰地看到边界，绘制抛物线边界曲线
    parabola_boundary = 1 + (p/q) * u_vec - (p^3 / q^3) * u_vec.^2;
    valid_boundary = (parabola_boundary > 1); % 只绘制 v>1 的部分
    plot(u_vec(valid_boundary), parabola_boundary(valid_boundary), 'b-', 'LineWidth', 2, 'DisplayName', 'v = 1 + (p/q)u - (p^3/q^3)u^2');
    plot(u_vec, ones(size(u_vec)), 'r--', 'LineWidth', 1.5, 'DisplayName', 'v = 1');

    % 设置图形属性
    title(sprintf('p = %.2f, q = %.2f\n区域: 1 < v \\leq 1 + (p/q)u - (p^3/q^3)u^2', p, q));
    xlabel('u');
    ylabel('v');
    axis equal;
    xlim([u_min, u_max]);
    ylim([min(V_mesh(:)), max(V_mesh(:))]);
    legend('show');
    grid on;
    hold off;

end

% --- 主程序入口 ---
% 提示用户输入 p 值
p_input = input('请输入 p 的值 (0 <= p <= 1): ');

% 调用绘图函数
plot_region(p_input);