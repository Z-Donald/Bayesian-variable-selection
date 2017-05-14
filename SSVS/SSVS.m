function [Beta, Sigma2, Gamma] = SSVS(X, y, num_update, num_burnin, sigma2, ...
                                  gamma, beta, c, tau, nu, lambda, omega, R)
% Stochastic Search Variable Selection for Linear regression
%
%%%%%%%%%
% Input:%
%%%%%%%%%
% X:          matrix(n, p), predict variable;
% y:          matrix(n, 1), respond variable;
% num_update: int, number of update;
% num_burnin: int, number of burn-in;
% sigma2:     float, initial value of random variable 'sigma^2';
% gamma:      matrix(p, 1), initial value of random variable 'gamma';
% beta:       matrix(p, 1), initial value of random variable 'beta';
% c:          float, initial value of constant 'c';
% tau:        matrix(p, 1), initial value of constant 'tau';
% nu:         variable, initial value of constant 'nu';
% lambda:     float, initial value of constant 'lambda';
% omega:      float, initial value of constant 'omega';
% R:          matrix(p, p), inistal value of matrix 'R'
% 
%%%%%%%%%%
% Output:%
%%%%%%%%%%
% Beta:   matrix(p, num_update - num_burin), samples from the posterior distribution of 'beta'
% Sigma2: matrix(p, num_update - num_burin), samples from the posterior distribution of 'sigma2'
% Gamma:  matrix(p, num_update - num_burin), samples from the posterior distribution of 'gamma'
% 
%%%%%%%%%%%%%
% Reference:%
%%%%%%%%%%%%%
% George E I, McCulloch R E. Variable selection via Gibbs sampling[J]. 
% Journal of the American Statistical Association, 1993, 88(423): 881-889.
%
%%%%%%%%%% 
% Author:%
%%%%%%%%%%
% Dom Tsang
%
% 2017/05/05


[n, p] = size(X);
Beta = zeros(p, num_update);
Sigma2 = zeros(1, num_update);
Gamma = zeros(p, num_update);
j = 1;
for i = 1 : num_update
    
% Note: The covariance matrix must be a symmetric semi-positive definite matrix

    %% generate beta from pdf(13)
    d = (1 - gamma) .* tau + gamma .* c .* tau;
    %% original formula: beta ~ MVN((inv(X' * X + sigma2 * D .^ (-2)) * X' * y, sigma2 * inv(X' * X + sigma2 * D .^ (-2)))
    tempA = X' * X /sigma2 + diag(1 ./ d) * R \ diag(1 ./ d);
	% compute the inverse matrix of tempA 
    [U, S, ~] = svd(tempA);
    AA = U * sqrt(inv(S));
    A = AA*AA';
	
	% generate beta
    beta = mvnrnd( A / sigma2 * X' * y, A)';
	
    %% generate sigma from pdf(14)
    sigma2 = invgamrnd((n + nu) / 2, (sum((y - X * beta) .^ 2)+ lambda * nu) / 2);
	
    %% generate gamma from pdf(15)
    a = normpdf(beta, zeros(p, 1), c ^ 2 .* tau .^ 2) .* omega;
    b = normpdf(beta, zeros(p, 1), tau .^ 2) .* (1 - omega);    
    gamma = binornd(1, a ./ (a + b), p, 1);
	
	
    if i > num_burnin
        Beta(:, j) = beta;
        Sigma2(j) = sigma2;
        Gamma(:, j) = gamma;
        j = j + 1;
    end
end
