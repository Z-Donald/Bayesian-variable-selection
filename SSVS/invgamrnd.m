function sample = invgamrnd(a, b, N, M)
if nargin == 2
    N = 1;
    M = 1;
end

sample = 1 ./ gamrnd(a, 1/(b+1e-6), N, M);
end