function exportWebStimuli(circOut, extOpts, webDir, ghDir)
% EXPORTWEBSTIMULI Saves each selected circular face stimulus as an
% alpha-masked, display-normalised PNG for both web_experiment/stimuli/
% and gh_pages/stimuli/.

thisDir = fileparts(mfilename('fullpath'));
if nargin < 3 || isempty(webDir)
    webDir = fullfile(thisDir, 'web_experiment');
end
if nargin < 4 || isempty(ghDir)
    ghDir = fullfile(thisDir, 'gh_pages');
end

stimOutDir = fullfile(webDir, 'stimuli');
ghStimDir  = fullfile(ghDir, 'stimuli');
if ~exist(stimOutDir, 'dir'), mkdir(stimOutDir); end
if ~exist(ghStimDir, 'dir'),  mkdir(ghStimDir);  end

sz = extOpts.imSize;
[gx, gy] = meshgrid(linspace(-1, 1, sz));
mask = exp(-((gx / extOpts.maskRx).^2 + (gy / extOpts.maskRy).^2).^4);
outSz = 256;
alphaMask = max(0, min(1, imresize(mask, [outSz, outSz])));

nSteps = height(circOut.selectedTable);
fprintf('Exporting %d web stimuli to %s and %s...\n', nSteps, stimOutDir, ghStimDir);

for k = 1:nSteps
    im = im2double(imread(circOut.selectedTable.FilePath(k)));
    if size(im, 3) == 3
        im = rgb2gray(im);
    end
    if size(im, 1) ~= sz || size(im, 2) ~= sz
        im = imresize(im, [sz, sz]);
    end

    imNorm = normaliseImage(im, extOpts.gamma, mask);
    imDisp = max(0, min(1, 0.5 + 0.5 * (imNorm / 2.8)));
    imOut  = uint8(round(255 * imresize(imDisp, [outSz, outSz])));

    fname = sprintf('stim_%02d.png', k);
    imwrite(imOut, fullfile(stimOutDir, fname), 'Alpha', alphaMask);
    imwrite(imOut, fullfile(ghStimDir, fname),  'Alpha', alphaMask);
end

fprintf('Exported %d alpha-masked PNG stimuli.\n', nSteps);
end
