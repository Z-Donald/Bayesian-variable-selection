function [Beta, Gamma, Sigma2] = KM(X, y, num_update, num_burnin, beta_0, D_0,gamma, p_j, alpha, eta)
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
% beta_0:     matrix(p, 1), initial value of random variable 'beta_0';
% eta:        variable, initial value of constant 'eta';
% alpha:      float, initial value of constant 'alpha';
% p_j:        float, initial value of constant 'p_j';
% D_0:        matrix(p, p), inistal value of matrix 'D'
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
% Dom Tsang & Yangtao Lin
%
% 2017/05/11



[n, p] = size(X);
Beta = zeros(p, num_update);
Gamma = zeros(p, num_update);
Sigma2 = zeros(1, num_update);
sigma2=invgamrnd(alpha/2, eta/2);
num=1;
newgamma=zeros(p,1);
for i = 1 : num_update
    X1=X.*repmat(gamma',n,1);
    beta1=(inv(D_0) + (1/sigma2)*(X1)'*(X1))\((D_0)\beta_0+(1/sigma2)*(X1)'*y);
    D=eye(p)/(eye(p)/(D_0) + (1/sigma2)*(X1)'*(X1));
    beta = mvnrnd( beta1, D)';
    v=gamma.*beta;
    for j=1:p
         v1=v;
         v2=v;
         v1(j)=beta(j);
         v2(j)=0;
         c_j=p_j*exp(-(1/(2*sigma2))*(y-X*v1)'*(y-X*v1));
         d_j=(1-p_j)*exp(-(1/(2*sigma2))*(y-X*v2)'*(y-X*v2));
         pj=c_j/(c_j+d_j);
         newgamma(j,1)=binornd(1,pj);
         v1=[];
         v2=[];
    end
    gamma=newgamma;
    sigma2=invgamrnd((alpha+n)/2,(eta+(y-X*v)'*(y-X*v))/2);
    if i > num_burnin
    Beta(:, num) = beta;
    Sigma2(num) = sigma2;
    Gamma(:, num) = gamma;
    num = num + 1;
    end
end