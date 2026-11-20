function exportWebsiteAssets(stimDir, ghDir, imSize, nFreqs, nOris, minFreq, maxFreq, biasOriDeg, biasOriBW, biasMinSF, biasMaxSF, extOpts, mdlBase, mdlBiased, circOut)
% EXPORTWEBSITEASSETS Generates exact MATLAB figures and results.json for gh_pages/:
%   1) figures/filter_examples.png      - Individual log-scaled (ori, sf) filters
%   2) figures/filter_bank_sums.png     - Sum of Base vs Horizontal-Biased banks
%   3) figures/within_between_faces.png - Masked within- vs between-identity exemplars
%   4) results.json                     - Live model metrics for index.html

thisDir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(stimDir),    stimDir    = fullfile(fileparts(thisDir), 'Men'); end
if nargin < 2 || isempty(ghDir),      ghDir      = fullfile(thisDir, 'gh_pages'); end
if nargin < 3 || isempty(imSize),     imSize     = 512; end
if nargin < 4 || isempty(nFreqs),     nFreqs     = 8; end
if nargin < 5 || isempty(nOris),      nOris      = 16; end
if nargin < 6 || isempty(minFreq),    minFreq    = 4; end
if nargin < 7 || isempty(maxFreq),    maxFreq    = imSize / 2; end
if nargin < 8 || isempty(biasOriDeg), biasOriDeg = 20; end
if nargin < 9 || isempty(biasOriBW),  biasOriBW  = 45; end
if nargin < 10 || isempty(biasMinSF), biasMinSF  = minFreq * 2; end
if nargin < 11 || isempty(biasMaxSF), biasMaxSF  = minFreq * 4; end
if nargin < 12 || isempty(extOpts)
    extOpts = struct('imSize', imSize, 'gamma', 2, 'maskRx', 0.72, 'maskRy', 0.88);
end

figDir = fullfile(ghDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end

%% 1. Build Filter Banks
oriBW = 180 / nOris;
[baseFilts, filtSteps, allOris] = createSFOriFilters(imSize, nFreqs, nOris, minFreq, maxFreq, oriBW);
[biasedFilts, extraFilts, biasMeta] = addHorizontalBiasFilters(baseFilts, biasOriDeg, biasOriBW, biasMinSF, biasMaxSF);

fAxis = (-imSize/2 : imSize/2 - 1);

%% 2. Figure A: Specific Log-Scaled (Ori, SF) Filter Examples (Frequency Domain)
fEx = figure('Visible', 'off', 'Color', [0.07 0.08 0.10], 'Position', [100, 100, 960, 560]);
exSpecs = {
    baseFilts(:, :, 1, 2), sprintf('Base: img %.0f (freq %+.0f), %.1f cyc', allOris(1), allOris(1)-90, filtSteps(2));
    baseFilts(:, :, 5, 4), sprintf('Base: img %.0f (freq %+.0f), %.1f cyc', allOris(5), allOris(5)-90, filtSteps(4));
    baseFilts(:, :, 9, 6), sprintf('Base: img %.0f (freq %+.0f), %.1f cyc', allOris(9), allOris(9)-90, filtSteps(6));
    extraFilts(:, :, 1),   sprintf('Biased: img %+.0f (freq %+.0f), %.1f cyc', biasMeta.imgOris(1), biasMeta.freqOris(1), biasMeta.sfSteps(1));
    extraFilts(:, :, 2),   sprintf('Biased: img %+.0f (freq %+.0f), %.1f cyc', biasMeta.imgOris(2), biasMeta.freqOris(2), biasMeta.sfSteps(1));
    extraFilts(:, :, 3),   sprintf('Biased: img %+.0f (freq %+.0f), %.1f cyc', biasMeta.imgOris(1), biasMeta.freqOris(1), biasMeta.sfSteps(2));
};
for k = 1:6
    ax = subplot(2, 3, k, 'Parent', fEx);
    imagesc(ax, fAxis, fAxis, fftshift(exSpecs{k, 1}), [0, 1]);
    colormap(ax, 'hot');
    axis(ax, 'image');
    set(ax, 'Color', 'k', 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], 'FontSize', 9);
    xlabel(ax, 'f_x (cyc/img)'); ylabel(ax, 'f_y (cyc/img)');
    title(ax, exSpecs{k, 2}, 'Color', 'w', 'FontSize', 10, 'FontWeight', 'normal');
end
exportgraphics(fEx, fullfile(figDir, 'filter_examples.png'), 'Resolution', 180, 'BackgroundColor', [0.07 0.08 0.10]);
close(fEx);

%% 3. Figure B: Sum of Base vs Horizontal-Biased Filter Banks
baseSum   = fftshift(sum(reshape(baseFilts, imSize, imSize, []), 3));
biasedSum = fftshift(sum(biasedFilts, 3));
maxSum    = max(biasedSum(:));

fSum = figure('Visible', 'off', 'Color', [0.07 0.08 0.10], 'Position', [100, 100, 920, 400]);
ax1 = subplot(1, 2, 1, 'Parent', fSum);
imagesc(ax1, fAxis, fAxis, baseSum, [0, maxSum]);
colormap(ax1, 'parula'); colorbar(ax1, 'Color', [0.85 0.85 0.85]);
axis(ax1, 'image');
set(ax1, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], 'FontSize', 9);
xlabel(ax1, 'f_x (cyc/img)'); ylabel(ax1, 'f_y (cyc/img)');
title(ax1, sprintf('Sum of Base Bank (%d filters)', nOris * nFreqs), ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'normal');

ax2 = subplot(1, 2, 2, 'Parent', fSum);
imagesc(ax2, fAxis, fAxis, biasedSum, [0, maxSum]);
colormap(ax2, 'parula'); colorbar(ax2, 'Color', [0.85 0.85 0.85]);
axis(ax2, 'image');
set(ax2, 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], 'FontSize', 9);
xlabel(ax2, 'f_x (cyc/img)'); ylabel(ax2, 'f_y (cyc/img)');
title(ax2, sprintf('Sum of Horizontal-Biased Bank (%d filters)', size(biasedFilts, 3)), ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'normal');

exportgraphics(fSum, fullfile(figDir, 'filter_bank_sums.png'), 'Resolution', 180, 'BackgroundColor', [0.07 0.08 0.10]);
close(fSum);

%% 4. Figure C: Within-Identity vs Between-Identity Masked Exemplars
files = dir(fullfile(stimDir, 'im*.png'));
names = arrayfun(@(f) string(f.name(10:end-4)), files);
[uNames, ~, idIdx] = unique(names, 'stable');
counts = accumarray(idIdx, 1);

multiIDs = find(counts >= 4);
withinID = multiIDs(1);
withinFiles = files(find(idIdx == withinID, 4, 'first'));

betweenIDs = multiIDs(2:5);
betweenFiles = arrayfun(@(id) files(find(idIdx == id, 1, 'first')), betweenIDs);

[gx, gy] = meshgrid(linspace(-1, 1, imSize));
mask = exp(-((gx / extOpts.maskRx).^2 + (gy / extOpts.maskRy).^2).^4);

fFaces = figure('Visible', 'off', 'Color', [0.5 0.5 0.5], 'Position', [100, 100, 760, 400]);
allExFiles = [withinFiles; betweenFiles];
for k = 1:8
    im = im2double(imread(fullfile(allExFiles(k).folder, allExFiles(k).name)));
    if size(im, 3) == 3, im = rgb2gray(im); end
    if size(im, 1) ~= imSize || size(im, 2) ~= imSize
        im = imresize(im, [imSize, imSize]);
    end
    imNorm = normaliseImage(im, extOpts.gamma, mask);
    imDisp = max(0, min(1, 0.5 + 0.5 * (imNorm / 2.8)));

    ax = subplot(2, 4, k, 'Parent', fFaces);
    imagesc(ax, imDisp, [0, 1]);
    colormap(ax, gray(256));
    axis(ax, 'image', 'off');
    if k == 1
        title(ax, sprintf('Within-identity (%s)', uNames(withinID)), 'Color', 'w', 'FontSize', 10, 'FontWeight', 'normal');
    elseif k == 5
        title(ax, 'Between-identity (4 actors)', 'Color', 'w', 'FontSize', 10, 'FontWeight', 'normal');
    end
end
exportgraphics(fFaces, fullfile(figDir, 'within_between_faces.png'), 'Resolution', 180, 'BackgroundColor', [0.5 0.5 0.5]);
close(fFaces);

%% 5. Write results.json if model outputs are supplied
if nargin >= 15 && ~isempty(mdlBase) && ~isempty(mdlBiased) && ~isempty(circOut)
    res = struct( ...
        'nTotalImages',   numel(files), ...
        'nExemplars',     sum(mdlBiased.fitMask), ...
        'nIdentities',    numel(mdlBiased.classes), ...
        'nOris',          nOris, ...
        'nFreqs',         nFreqs, ...
        'nBaseFilters',   nOris * nFreqs, ...
        'nBiasedFilters', size(biasedFilts, 3), ...
        'biasOriDeg',     biasOriDeg, ...
        'biasOriBW',      biasOriBW, ...
        'baseTop1',       100 * mdlBase.cvAcc, ...
        'baseTop1SD',     100 * std(mdlBase.foldAcc), ...
        'baseFolds',      100 * mdlBase.foldAcc(:)', ...
        'baseTop5',       100 * mdlBase.cvTop5, ...
        'biasTop1',       100 * mdlBiased.cvAcc, ...
        'biasTop1SD',     100 * std(mdlBiased.foldAcc), ...
        'biasFolds',      100 * mdlBiased.foldAcc(:)', ...
        'biasTop5',       100 * mdlBiased.cvTop5, ...
        'chanceTop1',     100 * mdlBiased.chanceAcc, ...
        'chanceTop5',     min(100, 500 * mdlBiased.chanceAcc), ...
        'entropyBits',    circOut.entropy_bits, ...
        'nSteps',         height(circOut.selectedTable));

    fid = fopen(fullfile(ghDir, 'results.json'), 'w');
    fprintf(fid, '%s', jsonencode(res, 'PrettyPrint', true));
    fclose(fid);
end

fprintf('Exported website figures and metadata to %s\n', ghDir);
end
