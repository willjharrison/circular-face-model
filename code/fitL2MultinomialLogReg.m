function mdlOut = fitL2MultinomialLogReg(X, metaTable, nOris, nFreqs, opts)
% FITL2MULTINOMIALLOGREG Fits L2-regularised multinomial logistic regression
% predicting face identity from frequency-space filter responses.
%
% Uses inverse-class-frequency sample weighting by default (opts.balanceClasses
% = true) so identities with fewer exemplars are not suppressed by database
% frequency priors.

if nargin < 5, opts = struct(); end
if ~isfield(opts, 'lambda'),         opts.lambda         = 1e-3; end
if ~isfield(opts, 'minExemplars'),   opts.minExemplars   = 5; end
if ~isfield(opts, 'kFolds'),         opts.kFolds         = 5; end
if ~isfield(opts, 'maxIter'),        opts.maxIter        = 500; end
if ~isfield(opts, 'tol'),            opts.tol            = 1e-6; end
if ~isfield(opts, 'foldID'),         opts.foldID         = []; end
if ~isfield(opts, 'balanceClasses'), opts.balanceClasses = true; end

tStart = tic;
fitMask = metaTable.nExemplars >= opts.minExemplars;
X_fit   = X(fitMask, :);
y_raw   = removecats(metaTable.Identity(fitMask));
cats    = categories(y_raw);
y_idx   = double(y_raw);
nCls    = numel(cats);
[N, K]  = size(X_fit);

fprintf('Fitting L2 multinomial logistic regression (N=%d exemplars, C=%d identities [>=%d photos], K=%d predictors, lambda=%.4g, balanced=%d)...\n', ...
    N, nCls, opts.minExemplars, K, opts.lambda, opts.balanceClasses);

%% 1. Stratified K-Fold Cross-Validation
cvAcc    = NaN;
cvBalAcc = NaN;
cvTop5   = NaN;
cvNLL    = NaN;
foldAcc  = [];
cvYTrue  = y_idx;
cvYPred  = [];
foldID   = opts.foldID;

if opts.kFolds >= 2
    tCV = tic;
    if isempty(foldID)
        foldID = makeStratifiedFolds(y_idx, nCls, opts.kFolds);
    end
    nActualFolds = max(foldID);
    predAll   = zeros(N, 1);
    top5Hit   = false(N, 1);
    nllAll    = zeros(N, 1);
    foldAcc   = zeros(nActualFolds, 1);

    for f = 1:nActualFolds
        tFold = tic;
        te = (foldID == f);
        tr = ~te;

        mu_f = mean(X_fit(tr, :), 1);
        sd_f = max(std(X_fit(tr, :), 0, 1), 1e-8);
        X_tr = (X_fit(tr, :) - mu_f) ./ sd_f;
        X_te = (X_fit(te, :) - mu_f) ./ sd_f;

        w_tr = makeSampleWeights(y_idx(tr), nCls, opts.balanceClasses);
        [W_f, b_f] = solveSoftmaxL2(X_tr, y_idx(tr), nCls, opts.lambda, opts.maxIter, opts.tol, w_tr);
        logits_te  = X_te * W_f + b_f;
        logits_te  = logits_te - max(logits_te, [], 2);
        P_te       = exp(logits_te) ./ sum(exp(logits_te), 2);

        [~, sortIdx] = sort(P_te, 2, 'descend');
        predFold = sortIdx(:, 1);

        teIdx = find(te);
        for jj = 1:numel(teIdx)
            nllAll(teIdx(jj)) = -log2(max(P_te(jj, y_idx(teIdx(jj))), 1e-15));
        end

        predAll(te) = predFold;
        top5Hit(te) = any(sortIdx(:, 1:min(5, nCls)) == y_idx(te), 2);
        foldAcc(f)  = mean(predFold == y_idx(te));
        fprintf('  Fold %d/%d: %.2f%% (%.2fs)\n', f, nActualFolds, 100 * foldAcc(f), toc(tFold));
    end
    cvAcc    = mean(predAll == y_idx);
    cvTop5   = mean(top5Hit);
    cvNLL    = mean(nllAll);
    cvYPred  = predAll;

    confMat  = accumarray([y_idx, predAll], 1, [nCls, nCls]);
    cvBalAcc = mean(diag(confMat) ./ max(sum(confMat, 2), 1));
    fprintf('  %d-fold CV Top-1: %.2f%% (Balanced: %.2f%%, SD=%.2f%%) | Top-5: %.2f%% | CV NLL: %.3f bits [%.2fs]\n', ...
        nActualFolds, 100 * cvAcc, 100 * cvBalAcc, 100 * std(foldAcc), 100 * cvTop5, cvNLL, toc(tCV));
end

%% 2. Fit Full Multi-Exemplar Model
tFull = tic;
mu = mean(X_fit, 1);
sd = max(std(X_fit, 0, 1), 1e-8);
X_fit_std = (X_fit - mu) ./ sd;

w_all = makeSampleWeights(y_idx, nCls, opts.balanceClasses);
[W, b, lossHist] = solveSoftmaxL2(X_fit_std, y_idx, nCls, opts.lambda, opts.maxIter, opts.tol, w_all);
fprintf('  Full model converged in %d iters (%.2fs).\n', numel(lossHist), toc(tFull));

% Information criteria penalising parameter count (K + 1) * C
logitsFull = X_fit_std * W + b;
logitsFull = logitsFull - max(logitsFull, [], 2);
P_full     = exp(logitsFull) ./ sum(exp(logitsFull), 2);
logLikFull = sum(log(max(P_full(sub2ind([N, nCls], (1:N)', y_idx)), 1e-15)));
nParams    = (K + 1) * nCls;
aic        = 2 * nParams - 2 * logLikFull;
bic        = nParams * log(N) - 2 * logLikFull;

%% 3. Project All Images into Top-2 Identity-Predictive Weight Subspace
[U, S, ~] = svd(W, 'econ');
W_2d = U(:, 1:2);
sv2  = diag(S(1:2, 1:2));

X_all_std = (X - mu) ./ sd;
Z = X_all_std * W_2d;

wVec = sqrt(sum(W.^2, 2));
nBase = nOris * nFreqs;
channelImportance = reshape(wVec(1:nBase), nOris, nFreqs);
if K > nBase
    extraImportance = wVec(nBase+1:end);
else
    extraImportance = [];
end

mdlOut = struct( ...
    'W',                 W, ...
    'b',                 b, ...
    'W_2d',              W_2d, ...
    'singularValues',    diag(S), ...
    'top2VarExplained',  sum(sv2.^2) / max(sum(diag(S).^2), eps), ...
    'Z',                 Z, ...
    'mu',                mu, ...
    'sd',                sd, ...
    'classes',           {cats}, ...
    'fitMask',           fitMask, ...
    'foldID',            foldID, ...
    'cvAcc',             cvAcc, ...
    'cvBalAcc',          cvBalAcc, ...
    'cvTop5',            cvTop5, ...
    'cvNLL',             cvNLL, ...
    'cvYTrue',           cvYTrue, ...
    'cvYPred',           cvYPred, ...
    'foldAcc',           foldAcc, ...
    'chanceAcc',         1 / nCls, ...
    'aic',               aic, ...
    'bic',               bic, ...
    'nParams',           nParams, ...
    'channelImportance', channelImportance, ...
    'extraImportance',   extraImportance, ...
    'lossHist',          lossHist, ...
    'lambda',            opts.lambda, ...
    'elapsedSec',        toc(tStart));
end

% =========================================================================
function w = makeSampleWeights(yIdx, nCls, balanceClasses)
N = numel(yIdx);
if ~balanceClasses
    w = ones(N, 1);
    return;
end
counts = accumarray(yIdx, 1, [nCls, 1]);
w = (N / nCls) ./ max(counts(yIdx), 1);
end

% =========================================================================
function [W, b, lossHist] = solveSoftmaxL2(X, yIdx, nCls, lambda, maxIter, tol, sampleWeights)
[N, K] = size(X);
Y_onehot = full(sparse(1:N, yIdx, 1, N, nCls));
sw = sampleWeights(:) / N;

W = zeros(K, nCls);
b = zeros(1, nCls);
W_y = W;
b_y = b;
t = 1;
L = 1.0;
lossHist = zeros(maxIter, 1);

for iter = 1:maxIter
    [loss_y, gW, gb] = softmaxLossGrad(X, Y_onehot, W_y, b_y, lambda, sw);

    while true
        W_next = W_y - (1 / L) * gW;
        b_next = b_y - (1 / L) * gb;
        loss_next = softmaxLossOnly(X, Y_onehot, W_next, b_next, lambda, sw);

        dW = W_next - W_y;
        db = b_next - b_y;
        quadApprox = loss_y + sum(gW(:) .* dW(:)) + sum(gb(:) .* db(:)) ...
            + 0.5 * L * (sum(dW(:).^2) + sum(db(:).^2));

        if loss_next <= quadApprox + 1e-12 || L > 1e8
            break;
        end
        L = L * 2;
    end

    L = max(1e-3, L * 0.9);
    t_next = 0.5 * (1 + sqrt(1 + 4 * t^2));
    beta_mom = (t - 1) / t_next;

    W_y = W_next + beta_mom * (W_next - W);
    b_y = b_next + beta_mom * (b_next - b);

    W = W_next;
    b = b_next;
    t = t_next;
    lossHist(iter) = loss_next;

    if iter > 5 && abs(lossHist(iter-1) - lossHist(iter)) < tol * max(1, lossHist(iter))
        lossHist = lossHist(1:iter);
        break;
    end
end
end

% =========================================================================
function [loss, gW, gb] = softmaxLossGrad(X, Y_onehot, W, b, lambda, sw)
logits = X * W + b;
logits = logits - max(logits, [], 2);
expL   = exp(logits);
P      = expL ./ sum(expL, 2);

ceLoss = -sum(sw .* sum(Y_onehot .* log(max(P, 1e-15)), 2));
l2Loss = 0.5 * lambda * sum(W(:).^2);
loss   = ceLoss + l2Loss;

diffP = sw .* (P - Y_onehot);
gW    = X' * diffP + lambda * W;
gb    = sum(diffP, 1);
end

% =========================================================================
function loss = softmaxLossOnly(X, Y_onehot, W, b, lambda, sw)
logits = X * W + b;
logits = logits - max(logits, [], 2);
expL   = exp(logits);
P      = expL ./ sum(expL, 2);
loss   = -sum(sw .* sum(Y_onehot .* log(max(P, 1e-15)), 2)) + 0.5 * lambda * sum(W(:).^2);
end

% =========================================================================
function foldID = makeStratifiedFolds(yIdx, nCls, kFolds)
foldID = zeros(numel(yIdx), 1);
for c = 1:nCls
    idx = find(yIdx == c);
    idx = idx(randperm(numel(idx)));
    for j = 1:numel(idx)
        foldID(idx(j)) = mod(j - 1, kFolds) + 1;
    end
end
end
