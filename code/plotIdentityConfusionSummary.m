function plotIdentityConfusionSummary(mdlBase, mdlBiased)
% PLOTIDENTITYCONFUSIONSUMMARY Prints a plain-English winner summary and
% plots the per-identity accuracy breakdown and C x C confusion matrix.

nCls = numel(mdlBiased.classes);
chancePct = 100 * mdlBiased.chanceAcc;

dTop1 = 100 * (mdlBiased.cvAcc  - mdlBase.cvAcc);
dTop5 = 100 * (mdlBiased.cvTop5 - mdlBase.cvTop5);
dNLL  = mdlBiased.cvNLL - mdlBase.cvNLL;
dBIC  = mdlBiased.bic   - mdlBase.bic;

fprintf('\n==================== PLAIN-ENGLISH MODEL SUMMARY ====================\n');
if dTop1 > 0
    fprintf('Best out-of-sample accuracy: Biased (+2x2 Horiz) by +%.2f%% Top-1 (%.2f%% vs %.2f%%)\n', ...
        dTop1, 100*mdlBiased.cvAcc, 100*mdlBase.cvAcc);
elseif dTop1 < 0
    fprintf('Best out-of-sample accuracy: Base (Complete) by +%.2f%% Top-1 (%.2f%% vs %.2f%%)\n', ...
        -dTop1, 100*mdlBase.cvAcc, 100*mdlBiased.cvAcc);
else
    fprintf('Out-of-sample Top-1 accuracy tied at %.2f%%.\n', 100*mdlBiased.cvAcc);
end
fprintf('  Top-5 accuracy difference : %+.2f%% (Biased %.2f%% vs Base %.2f%% | Chance %.2f%%)\n', ...
    dTop5, 100*mdlBiased.cvTop5, 100*mdlBase.cvTop5, min(100, 5*chancePct));
fprintf('  Held-out error (CV NLL)   : %+.3f bits (lower is better: Biased %.3f vs Base %.3f)\n', ...
    dNLL, mdlBiased.cvNLL, mdlBase.cvNLL);
fprintf('  Parameter-penalised BIC   : %+.0f (lower is better; %s wins on parsimony)\n', ...
    dBIC, ternaryStr(dBIC > 0, 'Base', 'Biased'));

if isempty(mdlBiased.cvYPred)
    fprintf('=====================================================================\n\n');
    return;
end

%% Per-Identity Accuracy & Confusion Matrix (using Biased model CV predictions)
yTrue = mdlBiased.cvYTrue;
yPred = mdlBiased.cvYPred;
confMat = accumarray([yTrue, yPred], 1, [nCls, nCls]);
nPerCls = sum(confMat, 2);
accPerCls = 100 * diag(confMat) ./ max(nPerCls, 1);

[accSorted, sortOrd] = sort(accPerCls, 'descend');
confNorm = confMat(sortOrd, sortOrd) ./ max(nPerCls(sortOrd), 1);

fprintf('\nTop 5 best-decoded identities:\n');
for i = 1:min(5, nCls)
    idx = sortOrd(i);
    fprintf('  %-22s : %5.1f%% (%d/%d correct)\n', ...
        char(mdlBiased.classes{idx}), accPerCls(idx), confMat(idx, idx), nPerCls(idx));
end
fprintf('Identities at 0%% accuracy  : %d / %d (Predicted >=1 time: %d / %d)\n', ...
    sum(accPerCls == 0), nCls, sum(sum(confMat, 1) > 0), nCls);
fprintf('=====================================================================\n\n');

%% Visualise Confusion Matrix & Per-Identity Accuracy
fig = figure('Name', 'Identity Accuracy & Confusions', 'Color', 'w', 'Position', [80, 80, 1180, 480]);

ax1 = subplot(1, 2, 1, 'Parent', fig);
imagesc(ax1, confNorm, [0, 0.5]);
colormap(ax1, 'hot');
colorbar(ax1);
axis(ax1, 'square');
xlabel(ax1, 'Predicted Identity (sorted by accuracy)');
ylabel(ax1, 'True Identity (sorted by accuracy)');
title(ax1, sprintf('Normalised Confusion Matrix (%d Identities)', nCls));

ax2 = subplot(1, 2, 2, 'Parent', fig);
bar(ax2, accSorted, 'FaceColor', [0.28, 0.52, 0.88], 'EdgeColor', 'none');
hold(ax2, 'on');
yline(ax2, 100*mdlBiased.cvAcc, 'k--', sprintf('Mean: %.1f%%', 100*mdlBiased.cvAcc), 'LineWidth', 1.4);
yline(ax2, chancePct, 'r--', sprintf('Chance: %.2f%%', chancePct), 'LineWidth', 1.2);
xlim(ax2, [0, nCls + 1]);
ylim(ax2, [0, max(100, max(accSorted) * 1.1)]);
grid(ax2, 'on'); box(ax2, 'off');
xlabel(ax2, 'Identity Rank (Best to Worst)');
ylabel(ax2, '5-Fold CV Accuracy (%)');
title(ax2, sprintf('Per-Identity Accuracy (%d/%d above chance)', sum(accPerCls > chancePct), nCls));

drawnow;
end

function out = ternaryStr(cond, a, b)
if cond, out = a; else, out = b; end
end
