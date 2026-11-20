%% RUN_CIRCULAR_FACE_MODEL
% End-to-end information-theoretic circular face space pipeline on Men/.
% Compares:
%   1) Base complete filter bank (nOris x nFreqs)
%   2) Horizontal-biased overcomplete bank (+ 2x2 low-SF horizontal filters)
% accounting for parameter count via CV accuracy, CV cross-entropy, and BIC/AIC.

clear;
close all;
clc;
tTotal = tic;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
faceImagesDir = fileparts(thisDir);
stimDir = fullfile(faceImagesDir, 'Men');

%% 1. Filter Bank Parameters (Base + Horizontal-Biased 2x2 Overcomplete)
tStep = tic;
imSize   = 512;
nFreqs   = 8;
nOris    = 32;
minFreq  = 2;           % cycles/image
maxFreq  = imSize / 2;  % cycles/image
oriBW    = 180 / nOris;

% Horizontal-bias parameters (2x2 extra filters around 0 deg horizontal)
biasOriDeg = oriBW;        % +/- deg around 0 deg horizontal
biasOriBW  = 45;        % narrow orientation bandwidth (deg)
biasMinSF  = minFreq*1;   % low-pass knot
biasMaxSF  = minFreq*4;        % upper cutoff of low-SF pair

[baseFilts, filtSteps, allOris] = createSFOriFilters(imSize, nFreqs, nOris, minFreq, maxFreq, oriBW);
[biasedFilts, ~, biasMeta]      = addHorizontalBiasFilters(baseFilts, biasOriDeg, biasOriBW, biasMinSF, biasMaxSF);

nBase   = nOris * nFreqs;
nBiased = size(biasedFilts, 3);
fprintf('[1/4] Filter banks built in %.2fs (Base: %d filters | Biased: %d filters).\n\n', ...
    toc(tStep), nBase, nBiased);

%% 2. Extract Global Summed Filter Responses (All Unedited Exemplars)
extOpts = struct('imSize', imSize, 'gamma', 2, 'maskRx', 0.72, 'maskRy', 0.88);
[X_biased, R, metaTable] = extractFilterBankEnergies(stimDir, biasedFilts, extOpts);
X_base = X_biased(:, 1:nBase);
fprintf('\n');

%% 3. Fit & Compare Base vs. Horizontal-Biased L2 Multinomial Models
rng(42); % identical stratified folds for fair paired comparison
glmOptsBase = struct('lambda', 1e-3, 'minExemplars', 5, 'kFolds', 5, 'maxIter', 500);
fprintf('--- Model A: Base Complete (%d filters) ---\n', nBase);
mdlBase = fitL2MultinomialLogReg(X_base, metaTable, nOris, nFreqs, glmOptsBase);

glmOptsBiased = glmOptsBase;
glmOptsBiased.foldID = mdlBase.foldID; % exact same CV splits
fprintf('\n--- Model B: Horizontal-Biased Overcomplete (%d filters) ---\n', nBiased);
mdlBiased = fitL2MultinomialLogReg(X_biased, metaTable, nOris, nFreqs, glmOptsBiased);

% Formal Model Comparison Table (accounting for extra 4*C parameters)
fprintf('\n=== Base vs. Horizontal-Biased Model Comparison ===\n');
compTable = table( ...
    ["Base (Complete)"; "Biased (+2x2 Horiz)"], ...
    [nBase; nBiased], ...
    [mdlBase.nParams; mdlBiased.nParams], ...
    100 * [mdlBase.cvAcc; mdlBiased.cvAcc], ...
    100 * [mdlBase.cvTop5; mdlBiased.cvTop5], ...
    [mdlBase.cvNLL; mdlBiased.cvNLL], ...
    [mdlBase.aic; mdlBiased.aic], ...
    [mdlBase.bic; mdlBiased.bic], ...
    'VariableNames', {'Model', 'nFilters', 'nParams', 'CV_Top1_Pct', 'CV_Top5_Pct', 'CV_NLL_Bits', 'AIC', 'BIC'});
disp(compTable);

fprintf('Extra horizontal filter importance (||W_k||_2):\n');
for k = 1:numel(biasMeta.labels)
    fprintf('  %-16s : %.3f\n', biasMeta.labels(k), mdlBiased.extraImportance(k));
end
plotIdentityConfusionSummary(mdlBase, mdlBiased);

%% 4. Define Circular Surprisal & Select CDF-Equalised Original Faces (Biased Model)
circOpts = struct('nSteps', 16, 'kappa', 6, 'rPctRange', [35, 70], 'uniqueIDs', true, 'doPlot', true, ...
    'imSize', imSize, 'gamma', extOpts.gamma, 'maskRx', extOpts.maskRx, 'maskRy', extOpts.maskRy);
circOut  = sampleEqualCircularSpace(mdlBiased.Z, metaTable, mdlBiased, allOris, filtSteps, circOpts);

%% 5. Export Web Stimuli & Save Results
disp(circOut.selectedTable(:, {'Filename', 'Identity', 'EqualStepAngleDeg', 'ActualAngleDeg', 'Radius', 'SurprisalBits'}));

exportWebStimuli(circOut, extOpts, fullfile(thisDir, 'web_experiment'), fullfile(thisDir, 'gh_pages'));
exportWebsiteAssets(stimDir, fullfile(thisDir, 'gh_pages'), imSize, nFreqs, nOris, minFreq, maxFreq, ...
    biasOriDeg, biasOriBW, biasMinSF, biasMaxSF, extOpts, mdlBase, mdlBiased, circOut);

savePath = fullfile(thisDir, 'circularFaceModel_results.mat');
save(savePath, 'X_base', 'X_biased', 'R', 'metaTable', 'mdlBase', 'mdlBiased', 'compTable', ...
    'circOut', 'allOris', 'filtSteps', 'biasMeta', '-v7.3');
fprintf('Saved results to %s\nTotal pipeline runtime: %.2fs.\n', savePath, toc(tTotal));
