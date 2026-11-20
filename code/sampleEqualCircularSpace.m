function circOut = sampleEqualCircularSpace(Z, metaTable, mdlOut, allOris, filtSteps, opts)
% SAMPLEEQUALCIRCULARSPACE Defines circular density & Shannon surprisal on the
% 2D identity-predictive space, selects original unedited images at
% CDF-equalised angular steps starting at 0 deg (due east), and plots
% diagnostics + alpha-blended circular face ring without identity names.

if nargin < 6, opts = struct(); end
if ~isfield(opts, 'nSteps'),    opts.nSteps    = 16; end
if ~isfield(opts, 'kappa'),     opts.kappa     = 6; end
if ~isfield(opts, 'rPctRange'), opts.rPctRange = [35, 70]; end
if ~isfield(opts, 'uniqueIDs'), opts.uniqueIDs = true; end
if ~isfield(opts, 'doPlot'),    opts.doPlot    = true; end
if ~isfield(opts, 'imSize'),    opts.imSize    = 256; end
if ~isfield(opts, 'gamma'),     opts.gamma     = 2; end
if ~isfield(opts, 'maskRx'),    opts.maskRx    = 0.72; end
if ~isfield(opts, 'maskRy'),    opts.maskRy    = 0.88; end

tStart = tic;
Z_c    = Z - median(Z, 1);
theta  = mod(atan2(Z_c(:, 2), Z_c(:, 1)), 2 * pi); % [0, 2*pi), 0 deg = due east
radius = hypot(Z_c(:, 1), Z_c(:, 2));

%% 1. Circular Density p(theta), Entropy H(Theta), and Surprisal I(theta)
nGrid = 720;
thetaGrid = linspace(0, 2 * pi, nGrid + 1);
thetaGrid(end) = [];
dTheta = 2 * pi / nGrid;

kappa = opts.kappa;
p_grid = mean(exp(kappa * cos(thetaGrid - theta)), 1) / (2 * pi * besseli(0, kappa));
p_grid = p_grid / (sum(p_grid) * dTheta);

surprisal_grid = -log2(max(p_grid, eps));
entropy_bits   = -sum(p_grid .* log2(max(p_grid, eps))) * dTheta;

p_img         = interp1([thetaGrid, 2*pi], [p_grid, p_grid(1)], theta, 'linear');
surprisal_img = -log2(max(p_img, eps));

%% 2. CDF Equalisation Starting at 0 deg (Due East)
cdf_grid = cumsum(p_grid) * dTheta;
cdf_grid = cdf_grid / cdf_grid(end);

[cdfUnique, uIdx] = unique([0, cdf_grid], 'stable');
thetaAll = [0, thetaGrid];
thetaUnique = thetaAll(uIdx);

targetCDF    = linspace(0, 1 - 1/opts.nSteps, opts.nSteps); % starts at 0 -> 0 deg (due east)
targetAngles = interp1(cdfUnique, thetaUnique, targetCDF, 'linear');

%% 3. Select Original Unedited Exemplars in Annulus Nearest Target Angles
rLow   = prctile(radius, opts.rPctRange(1));
rHigh  = prctile(radius, opts.rPctRange(2));
inRing = (radius >= rLow) & (radius <= rHigh);
candIdx = find(inRing);

selectedIdx = zeros(opts.nSteps, 1);
usedIDs = [];

for k = 1:opts.nSteps
    avail = candIdx;
    if opts.uniqueIDs && ~isempty(usedIDs)
        avail = candIdx(~ismember(metaTable.IDIdx(candIdx), usedIDs));
    end
    if isempty(avail)
        avail = candIdx;
    end

    angDiff = abs(angle(exp(1i * (theta(avail) - targetAngles(k)))));
    [~, best] = min(angDiff);
    selectedIdx(k) = avail(best);
    usedIDs(end+1) = metaTable.IDIdx(selectedIdx(k)); %#ok<AGROW>
end

selectedTable = metaTable(selectedIdx, :);
selectedTable.EqualStepAngleDeg = linspace(0, 360 - 360/opts.nSteps, opts.nSteps)';
selectedTable.TargetAngleDeg    = rad2deg(targetAngles(:));
selectedTable.ActualAngleDeg    = rad2deg(theta(selectedIdx));
selectedTable.Radius            = radius(selectedIdx);
selectedTable.SurprisalBits     = surprisal_img(selectedIdx);

circOut = struct( ...
    'Z_centered',     Z_c, ...
    'theta',          theta, ...
    'radius',         radius, ...
    'thetaGrid',      thetaGrid, ...
    'p_grid',         p_grid, ...
    'surprisal_grid', surprisal_grid, ...
    'entropy_bits',   entropy_bits, ...
    'cdf_grid',       cdf_grid, ...
    'p_img',          p_img, ...
    'surprisal_img',  surprisal_img, ...
    'targetAngles',   targetAngles, ...
    'selectedIdx',    selectedIdx, ...
    'selectedTable',  selectedTable, ...
    'elapsedSec',     toc(tStart));

fprintf('Circular sampling complete in %.2fs (H = %.2f bits).\n', circOut.elapsedSec, entropy_bits);

if opts.doPlot
    plotCircularSpaceDiagnostics(circOut, mdlOut, allOris, filtSteps);
    plotCircularFaceRing(circOut, opts);
end
end

% =========================================================================
function plotCircularSpaceDiagnostics(circOut, mdlOut, allOris, filtSteps)
fig1 = figure('Name', 'Circular Face Space Diagnostics', 'Color', 'w', 'Position', [40, 80, 1250, 400]);

ax1 = subplot(1, 3, 1, 'Parent', fig1);
scatter(ax1, circOut.Z_centered(:,1), circOut.Z_centered(:,2), 14, circOut.surprisal_img, 'filled', 'MarkerFaceAlpha', 0.45);
hold(ax1, 'on');
sel = circOut.selectedIdx;
plot(ax1, circOut.Z_centered(sel, 1), circOut.Z_centered(sel, 2), 'ko-', 'LineWidth', 1.4, 'MarkerFaceColor', 'r', 'MarkerSize', 6);
colorbar(ax1); colormap(ax1, 'parula');
axis(ax1, 'equal'); grid(ax1, 'on'); box(ax1, 'off');
xlabel(ax1, 'Weight Axis 1'); ylabel(ax1, 'Weight Axis 2');
title(ax1, sprintf('2D Space (Top-1: %.1f%%, Top-5: %.1f%%)', 100*mdlOut.cvAcc, 100*mdlOut.cvTop5));

ax2 = subplot(1, 3, 2, 'Parent', fig1);
angDeg = rad2deg(circOut.thetaGrid);
yyaxis(ax2, 'left');
plot(ax2, angDeg, circOut.p_grid, 'LineWidth', 1.8);
ylabel(ax2, 'Density p(\theta)');
yyaxis(ax2, 'right');
plot(ax2, angDeg, circOut.surprisal_grid, 'LineWidth', 1.8);
ylabel(ax2, 'Surprisal I(\theta) [bits]');
hold(ax2, 'on');
for a = rad2deg(circOut.targetAngles)
    xline(ax2, a, '--', 'Alpha', 0.35);
end
xlim(ax2, [0, 360]); grid(ax2, 'on'); box(ax2, 'off');
xlabel(ax2, 'Angle \theta (deg, 0^\circ = East)');
title(ax2, sprintf('Circular Density & Surprisal (H = %.2f bits)', circOut.entropy_bits));

ax3 = subplot(1, 3, 3, 'Parent', fig1);
imagesc(ax3, mdlOut.channelImportance);
colorbar(ax3); colormap(ax3, 'hot');
set(ax3, 'TickDir', 'out', 'Box', 'off', ...
    'XTick', 1:numel(filtSteps), 'XTickLabel', round(filtSteps, 1), ...
    'YTick', 1:numel(allOris), 'YTickLabel', round(allOris));
xlabel(ax3, 'Spatial Frequency (cyc/img)');
ylabel(ax3, 'Orientation (deg)');
title(ax3, 'Base Filter Importance ||W_k||_2');

drawnow;
end

% =========================================================================
function plotCircularFaceRing(circOut, opts)
% Places the selected original faces around the circumference of a circle at
% equal perceptual steps (0:22.5:337.5), normalised to [0, 1] for display and
% alpha-blended with the elliptical mask so transparent corners never overlap.
nSteps = height(circOut.selectedTable);
fig2 = figure('Name', 'Circular Face Ring (Perceptual Inspection)', 'Color', [0.5 0.5 0.5], 'Position', [120, 60, 860, 860]);

axBg = axes('Parent', fig2, 'Position', [0.05, 0.05, 0.90, 0.90], 'Color', [0.5 0.5 0.5]);
hold(axBg, 'on');
tCirc = linspace(0, 2*pi, 360);
plot(axBg, cos(tCirc), sin(tCirc), '--', 'Color', [0.85 0.85 0.85], 'LineWidth', 1.2);
axis(axBg, 'equal');
xlim(axBg, [-1.45, 1.45]);
ylim(axBg, [-1.45, 1.45]);
axis(axBg, 'off');
title(axBg, sprintf('%d Masked Faces Around Equal-Step Circle (0^\\circ = East, H = %.2f bits)', ...
    nSteps, circOut.entropy_bits), 'FontSize', 13, 'FontWeight', 'normal', 'Color', 'w');

% Build exact same analysis mask & alpha channel
sz = opts.imSize;
[gx, gy] = meshgrid(linspace(-1, 1, sz));
mask = exp(-((gx / opts.maskRx).^2 + (gy / opts.maskRy).^2).^4);
alphaThumb = imresize(mask, [128, 128]);
alphaThumb = max(0, min(1, alphaThumb));

thumbFrac = min(0.17, 0.78 * pi / max(nSteps, 6));
ringRadiusNorm = 0.34;

for k = 1:nSteps
    angRad = deg2rad(circOut.selectedTable.EqualStepAngleDeg(k));
    cx = 0.5 + ringRadiusNorm * cos(angRad);
    cy = 0.5 + ringRadiusNorm * sin(angRad);

    im = im2double(imread(circOut.selectedTable.FilePath(k)));
    if size(im, 3) == 3, im = rgb2gray(im); end
    if size(im, 1) ~= sz || size(im, 2) ~= sz
        im = imresize(im, [sz, sz]);
    end
    imNorm  = normaliseImage(im, opts.gamma, mask);
    % Map z-scored [-2.8, +2.8] SDs cleanly to [0, 1] display luminance (0 -> 0.5)
    imDisp  = max(0, min(1, 0.5 + 0.5 * (imNorm / 2.8)));
    imThumb = imresize(imDisp, [128, 128]);

    axIm = axes('Parent', fig2, 'Position', [cx - thumbFrac/2, cy - thumbFrac/2, thumbFrac, thumbFrac], ...
        'Color', 'none');
    hIm = imagesc(axIm, imThumb, [0, 1]);
    set(hIm, 'AlphaData', alphaThumb);
    colormap(axIm, gray(256));
    axis(axIm, 'image', 'off');
    title(axIm, sprintf('%.1f^\\circ', circOut.selectedTable.EqualStepAngleDeg(k)), ...
        'FontSize', 8, 'FontWeight', 'normal', 'Color', 'w');
end

drawnow;
end
