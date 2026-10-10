n = input('请输入变量个数n(大于等于5的整数)：');
if n < 5 || mod(n,1) ~= 0
    error('n必须是大于等于5的整数。')
end

k=input('请输入变量最小值与最大值的比值k(0<k<1)：');
if k<=0 || k>1
    error('k必须满足0<k<1。')
end

syms p q;
A = sym(ones(n));  % 初始化为符号全1矩阵

% 处理所有 i -> i+1 和 i -> i+2 的连接（循环）
for i = 1:n

    A(i,i)=A(i,i)+k^2 * p^2;

    % i+1 (mod n)
    j1 = mod(i, n) + 1;
    % i+2 (mod n)
    j2 = mod(i+1, n) + 1;
    
    % 减去 n*p * x_i x_{i+1} → 对称地减 (n/2)*p 到两个位置
    A(i, j1) = A(i, j1) - (n/2)*p + (1/2)*p*(1-2*p)*k*k;
    A(j1, i) = A(j1, i) - (n/2)*p + (1/2)*p*(1-2*p)*k*k;
    
    % 减去 n*q * x_i x_{i+2} → 对称地减 (n/2)*q
    A(i, j2) = A(i, j2) - (n/2)*q - (1/2)*p*(1-p)*k*k;
    A(j2, i) = A(j2, i) - (n/2)*q - (1/2)*p*(1-p)*k*k;
end

% 显示结果
disp('二次型对应的矩阵 A 为：');
pretty(A)

% Preallocate cell to store results (optional)
D = sym(zeros(1,n));

% Compute sum of all principal minors of order 1 to n
fprintf('Computing sum of all principal minors...\n\n');
for k = 1:n
    idx_list = nchoosek(1:n, k);   % All index combinations of size k
    Dk = sym(0);                   % Initialize sum of k-th order principal minors
    
    for i = 1:size(idx_list, 1)
        idx = idx_list(i, :);
        Ak = A(idx, idx);          % k-th order principal submatrix (not just leading)
        Dk = Dk + simplify(det(Ak)); % accumulate determinant
    end
    
    Dk = simplify(Dk);             % final simplification
    D(k) = Dk;                     % store in D(k), as before
    
    fprintf('Order %d:\n', k);
    Dk_factored = factor(Dk);
    fprintf('Order %d (factored):\n', k);
    pretty(Dk_factored)
    fprintf('\n');
end

% ========== 新增部分：代入 q = 1 - p 并求 p ∈ [0,1] 中 Dk(p) >= 0 的范围 ==========
fprintf('\n=== 代入 q = 1 - p 后，求各阶顺序主子式 ≥ 0 的 p 范围 (p ∈ [0,1]) ===\n\n');

p_sym = p;  % 确保使用符号 p
results = cell(n, 1);  % 存储每个 k 的解区间

for k = 1:n
    fprintf('处理第 %d 阶主子式之和...\n', k);
    
    % 1. 代入 q = 1 - p
    Dk_p = subs(D(k), q, 1 - p);
    Dk_p = simplify(Dk_p);
    
    % 2. 如果是常数（如 k=1 时可能为 1）
    if isAlways(Dk_p == 0)
        fprintf('  恒等于 0 → 所有 p ∈ [0,1] 满足 ≥ 0\n');
        results{k} = [0, 1];
        continue;
    elseif isAlways(Dk_p > 0)
        fprintf('  恒正 → 所有 p ∈ [0,1] 满足 ≥ 0\n');
        results{k} = [0, 1];
        continue;
    elseif isAlways(Dk_p < 0)
        fprintf('  恒负 → 无解\n');
        results{k} = [];
        continue;
    end
    
    % 3. 尝试求实根（在实数范围内）
    try
        roots_sym = solve(Dk_p == 0, p, 'Real', true);
        % 转换为数值（去重、排序）
        roots_num = double(roots_sym);
        % 过滤复数（理论上不应有，但以防万一）
        roots_real = roots_num(imag(roots_num) == 0);
        roots_real = real(roots_real);
        % 保留 [0,1] 内的根
        roots_in_interval = roots_real(roots_real >= 0 & roots_real <= 1);
        % 去重并排序
        roots_unique = unique(roots_in_interval);
    catch
        % 若符号求根失败，尝试数值求根（使用 vpasolve 多起点）
        fprintf('  符号求根失败，尝试数值方法...\n');
        roots_unique = [];
        % 在 [0,1] 内多点采样找变号区间
        f = matlabFunction(Dk_p, 'Vars', p);
        xx = linspace(0, 1, 1001);
        yy = arrayfun(f, xx);
        % 找符号变化点（粗略）
        sign_changes = find(yy(1:end-1) .* yy(2:end) <= 0);
        for idx = sign_changes
            try
                r = vpasolve(Dk_p == 0, p, [xx(idx), xx(idx+1)]);
                r_num = double(r);
                if r_num >= 0 && r_num <= 1
                    roots_unique = [roots_unique; r_num];
                end
            catch
                continue;
            end
        end
        roots_unique = unique(roots_unique);
    end
    
    % 4. 构建测试区间：[0, r1, r2, ..., 1]
    test_points = [0; roots_unique(:); 1];
    test_points = unique(test_points);
    test_points = sort(test_points);
    
    % 5. 在每个小区间中点测试符号
    valid_intervals = [];
    for i = 1:length(test_points)-1
        left = test_points(i);
        right = test_points(i+1);
        mid = (left + right) / 2;
        
        % 避免在根上测试（但中点一般不是根）
        val = double(subs(Dk_p, p, mid));
        if isnan(val) || isinf(val)
            val = 0; % 边界情况
        end
        
        if val >= -1e-10  % 允许数值误差
            % 合并到有效区间
            if isempty(valid_intervals)
                valid_intervals = [left, right];
            else
                % 检查是否可合并
                if left <= valid_intervals(end,2)
                    valid_intervals(end,2) = right;
                else
                    valid_intervals = [valid_intervals; left, right];
                end
            end
        end
    end
    
    % 6. 特殊处理：如果多项式在端点为0，应包含端点（已包含）
    results{k} = valid_intervals;
    
    % 7. 输出结果
    if isempty(valid_intervals)
        fprintf('  无解\n');
    else
        fprintf('  解集为：\n');
        for j = 1:size(valid_intervals,1)
            fprintf('    [%.6f, %.6f]\n', valid_intervals(j,1), valid_intervals(j,2));
        end
    end
    fprintf('\n');
end

% ========== 可选：输出所有结果汇总 ==========
fprintf('\n=== 最终结果汇总 ===\n');
for k = 1:n
    fprintf('k = %d: ', k);
    if isempty(results{k})
        fprintf('无解\n');
    else
        intervals = results{k};
        str_parts = cell(size(intervals,1),1);
        for j = 1:size(intervals,1)
            str_parts{j} = sprintf('[%.4f, %.4f]', intervals(j,1), intervals(j,2));
        end
        fprintf('%s\n', strjoin(str_parts, ' ∪ '));
    end
end