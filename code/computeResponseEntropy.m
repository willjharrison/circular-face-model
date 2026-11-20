function [H_pop, H_chan] = computeResponseEntropy(R, nBins)
% [H_pop, H_chan] = computeResponseEntropy(R, nBins)
%
% Computes Shannon entropy (in bits) of filter-bank responses using base MATLAB.

if ndims(R) == 4 && size(R, 1) == 1 && size(R, 2) == 1
    R = squeeze(R);
end

[nOris, nFreqs, N] = size(R);
K = nOris * nFreqs;

R_mat = reshape(max(R, 0), K, N);

% 1. Per-input population entropy across all (nOris * nFreqs) filters
P = bsxfun(@rdivide, R_mat, max(sum(R_mat, 1), eps));
H_pop = -sum(P .* log2(max(P, eps)), 1);

% 2. Across-input entropy per filter channel (when N > 1 scenes are passed)
if N < 2
    H_chan = [];
    return
end

if nargin < 2 || isempty(nBins)
    nBins = max(5, round(sqrt(N)));
end

rMin = min(R_mat, [], 2);
rRange = max(R_mat, [], 2) - rMin;
edges = linspace(0, 1, nBins + 1);
H_vec = zeros(K, 1);

for k = 1:K
    if rRange(k) > 0
        counts = histcounts((R_mat(k, :) - rMin(k)) / rRange(k), edges);
        pk = counts(counts > 0) / N;
        H_vec(k) = -sum(pk .* log2(pk));
    end
end

H_chan = reshape(H_vec, nOris, nFreqs);
end
