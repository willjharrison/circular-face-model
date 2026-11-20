function [X, R, metaTable] = extractFilterBankEnergies(stimDir, allFilts, opts)
% EXTRACTFILTERBANKENERGIES Extracts summed frequency-space filter responses
% from unedited face images without averaging across exemplars or spatial pooling.
%
% Accepts either 4D [H x W x nOris x nFreqs] or 3D [H x W x nFilters] allFilts.

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'imSize'),   opts.imSize   = size(allFilts, 1); end
if ~isfield(opts, 'gamma'),    opts.gamma    = 2; end
if ~isfield(opts, 'maskRx'),   opts.maskRx   = 0.72; end
if ~isfield(opts, 'maskRy'),   opts.maskRy   = 0.88; end
if ~isfield(opts, 'showLive'), opts.showLive = true; end

tStart = tic;
files = dir(fullfile(stimDir, 'im*.png'));
if isempty(files)
    error('No im*.png files found in %s', stimDir);
end

nImages = numel(files);
sz = opts.imSize;
if ndims(allFilts) == 4
    [~, ~, nOris, nFreqs] = size(allFilts);
    filtsFlat = reshape(allFilts, sz, sz, []);
    K = nOris * nFreqs;
else
    K = size(allFilts, 3);
    filtsFlat = allFilts;
end

% Smooth super-Gaussian elliptical mask to suppress background/border artifacts
[gx, gy] = meshgrid(linspace(-1, 1, sz));
mask = exp(-((gx / opts.maskRx).^2 + (gy / opts.maskRy).^2).^4);

X = zeros(nImages, K);
R = zeros(K, nImages);
filenames  = strings(nImages, 1);
filepaths  = strings(nImages, 1);
identities = strings(nImages, 1);

% Circular buffer for last 10 masked & normalised thumbnails
thumbSz     = 96;
last10Imgs  = cell(10, 1);
last10Names = strings(10, 1);
if opts.showLive
    hLiveFig = figure('Name', 'Extraction Preview (Last 10 Masked Faces)', ...
        'Color', 'w', 'Position', [80, 120, 900, 380]);
end

fprintf('Extracting %d frequency-space filter responses from %d images...\n', K, nImages);

for i = 1:nImages
    fname = files(i).name;
    filenames(i)  = string(fname);
    filepaths(i)  = string(fullfile(files(i).folder, fname));
    identities(i) = string(fname(10:end-4)); % matches getPhotoInfo.m convention

    im = im2double(imread(filepaths(i)));
    if size(im, 3) == 3
        im = rgb2gray(im);
    end
    if size(im, 1) ~= sz || size(im, 2) ~= sz
        im = imresize(im, [sz, sz]);
    end

    imNorm = normaliseImage(im, opts.gamma, mask);
    F = fft2(imNorm);
    F(1, 1) = 0; % zero DC directly in Fourier space without shifting masked background

    % Vectorised filter-bank response and global sum per filter
    filtResp = abs(ifft2(F .* filtsFlat));
    R_im     = squeeze(sum(filtResp, [1 2])); % [K x 1]

    R(:, i) = R_im(:);
    X(i, :) = log(R_im(:)' + eps);

    % Store thumbnail in rolling buffer of last 10 faces
    bufIdx = mod(i - 1, 10) + 1;
    last10Imgs{bufIdx}  = imresize(imNorm, [thumbSz, thumbSz]);
    last10Names(bufIdx) = identities(i);

    if mod(i, 250) == 0 || i == nImages
        fprintf('  Processed %d / %d images (%.2fs elapsed)\n', i, nImages, toc(tStart));
        if opts.showLive && isvalid(hLiveFig)
            nShow = min(i, 10);
            order = mod((i - nShow):(i - 1), 10) + 1;
            for k = 1:nShow
                ax = subplot(2, 5, k, 'Parent', hLiveFig);
                imagesc(ax, last10Imgs{order(k)}, [-2.8, 2.8]);
                colormap(ax, gray(256));
                axis(ax, 'image', 'off');
                title(ax, char(last10Names(order(k))), 'FontSize', 8, 'FontWeight', 'normal');
            end
            sgtitle(hLiveFig, sprintf('Last 10 Masked & Normalised Faces (%d / %d, %.1fs)', ...
                i, nImages, toc(tStart)), 'FontSize', 11, 'FontWeight', 'normal');
            drawnow;
        end
    end
end

[~, ~, idIdx] = unique(identities, 'stable');
counts = accumarray(idIdx, 1);
nExemplars = counts(idIdx);

metaTable = table(filenames, filepaths, categorical(identities), idIdx, nExemplars, ...
    'VariableNames', {'Filename', 'FilePath', 'Identity', 'IDIdx', 'nExemplars'});
fprintf('Feature extraction complete in %.2fs.\n', toc(tStart));
end
