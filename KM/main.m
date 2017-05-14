clc
clear
%% load data
n = 60;
p = 5;
X = mvnrnd(zeros(n, 1), eye(n), p)';
% X(:, 3) = X(:, 5) + 0.15 .* normrnd(0, 1, n, 1);
epsilon = mvnrnd(zeros(n, 1),eye(n))';
y = X(:, 4) + 1.2 * X(:, 5) +2.5.*epsilon;

%% set up
num_update = 8000;
num_burnin = 200;

%% initialization
beta_0=zeros(p,1);
D_0=16*eye(p);
gamma=ones(p,1);
p_j=1/2;
alpha=0.01;
eta=0.01;
%% Gibbs sampling
[Beta, Gamma, Sigma] = KM(X, y, num_update, num_burnin, beta_0, D_0,gamma, p_j, alpha, eta);

%% model quantization
m = zeros(1, num_update);
for i = 1 : p
    m = m + Gamma(i, :) * 2^(i - 1);
end
tab = tabulate(m);