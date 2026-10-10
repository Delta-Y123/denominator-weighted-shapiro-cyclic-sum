% Define symbolic variables
syms p q;

% Construct the 6x6 matrix
A = [
    1,           1 - 3*p,   1 - 3*q,   1,           1 - 3*q,   1 - 3*p;
    1 - 3*p,     1,         1 - 3*p,   1 - 3*q,     1,         1 - 3*q;
    1 - 3*q,     1 - 3*p,   1,         1 - 3*p,     1 - 3*q,   1;
    1,           1 - 3*q,   1 - 3*p,   1,           1 - 3*p,   1 - 3*q;
    1 - 3*q,     1,         1 - 3*q,   1 - 3*p,     1,         1 - 3*p;
    1 - 3*p,     1 - 3*q,   1,         1 - 3*q,     1 - 3*p,   1
];

% Display the matrix
disp('Matrix A:');
pretty(A)


% Preallocate cell to store results (optional)
D = sym(zeros(1,6));

% Compute leading principal minors of order 1 to 6
fprintf('Computing leading principal minors...\n\n');
for k = 1:6
    Ak = A(1:k, 1:k);          % k-th leading principal submatrix
    Dk = simplify(det(Ak));    % determinant (minor)
    D(k) = Dk;
    
    fprintf('Order %d:\n', k);
    Dk_factored = factor(Dk);
    fprintf('Order %d (factored):\n', k);
    pretty(Dk_factored)
    fprintf('\n');
end

% Optional: export results to workspace
% D1 = D(1); D2 = D(2); ... D6 = D(6);