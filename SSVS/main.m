%% simulated data
n = 60;
p = 5;
X = mvnrnd(zeros(n, 1), eye(n), p)';
epsilon = mvnrnd(zeros(n, 1), 2.5 .* eye(n))';
y = X(:, 4) + 1.2 * X(:, 5) + epsilon;
beta_ls = X\y;

%% set up
num_update = 5000;
num_burnin = 200;

%% initialization
c = 10;
tau = ones(p, 1) .* 0.33;
nu = 2e-4;
lambda = 1;
omega = 0.5;
R = eye(p);
gamma_0 = binornd(1, omega, p, 1);
D = diag((1 - gamma_0) .* tau + gamma_0 .* c .* tau);
%using the result of least squared to initialize
beta_0 = beta_ls;
sigma2_0 = invgamrnd(nu / 2, lambda * nu / 2);


%% Gibbs sampling
[Beta, Sigma2, Gamma] = SSVS(X, y, num_update, num_burnin, sigma2_0, gamma_0, beta_0, c, tau, nu, lambda, omega, R);

%% model quantization
m = zeros(1, num_update);
for i = 1 : p
    m = m + Gamma(i, :) * 2^(i - 1);
end
tab = tabulate(m);