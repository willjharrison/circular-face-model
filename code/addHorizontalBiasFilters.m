function [combinedFilts, extraFilts, meta] = addHorizontalBiasFilters(baseFilts, biasOriDeg, biasOriBW, minSF, maxLowSF)
% ADDHORIZONTALBIASFILTERS Appends 2x2 horizontal-biased filters (2 low-SF
% bands whose sum is low-pass x 2 orientations at +/- biasOriDeg around
% horizontal image features, i.e. +/- 90 deg in frequency space).
%
% Inputs:
%   baseFilts  - [H x W x nOris x nFreqs] or [H x W x nBase] FFT filter bank
%   biasOriDeg - Offset from image-space horizontal (0 deg) in deg (default: 10)
%   biasOriBW  - Orientation half-bandwidth in deg (default: 10)
%   minSF      - Lowest SF knot in cycles/image (default: 4)
%   maxLowSF   - Upper cutoff of the low-pass pair in cycles/image (default: 16)
%
% Outputs:
%   combinedFilts - [H x W x (nBase + 4)] FFT filter bank
%   extraFilts    - [H x W x 4] the 4 added horizontal-biased filters
%   meta          - Struct describing the 4 added filters

if nargin < 2 || isempty(biasOriDeg), biasOriDeg = 10; end
if nargin < 3 || isempty(biasOriBW),  biasOriBW  = 10; end
if nargin < 4 || isempty(minSF),      minSF      = 4;  end
if nargin < 5 || isempty(maxLowSF),   maxLowSF   = 16; end

sz = size(baseFilts, 1);
if ndims(baseFilts) == 4
    baseFlat = reshape(baseFilts, sz, sz, []);
else
    baseFlat = baseFilts;
end

[angDist, radDist] = polarDistFun(sz);

[sfFilts, sfSteps] = createBandPassFilters_customSizes(radDist, 3, minSF, maxLowSF);
lowSFFilts = sfFilts(:, :, 1:2); % [low-pass, low-SF bandpass]

inHalfPlane = (angDist >= -90) & (angDist < 90);

% Horizontal features in image space correspond to +/- 90 deg (vertical midline)
% in 2D Fourier frequency space, matching createSFOriFilters (ori - 90).
imgOris  = [-biasOriDeg, +biasOriDeg];
freqOris = mod(imgOris - 90 + 90, 180) - 90;

extraFilts = zeros(sz, sz, 4);
labels = strings(4, 1);
idx = 0;

for sf = 1:2
    sfSlice = lowSFFilts(:, :, sf) .* inHalfPlane;
    for oi = 1:2
        idx = idx + 1;
        thisOri = freqOris(oi);
        dAng    = mod(angDist - thisOri + 90, 180) - 90;
        oriFilt = 0.5 * (1 + cos(pi * min(abs(dAng) / biasOriBW, 1)));
        extraFilts(:, :, idx) = fftshift(sfSlice .* oriFilt);
        labels(idx) = sprintf('SF%.1f_ImgOri%+.0f', sfSteps(sf), imgOris(oi));
    end
end

combinedFilts = cat(3, baseFlat, extraFilts);
meta = struct('sfSteps', sfSteps(1:2), 'imgOris', imgOris, 'freqOris', freqOris, ...
    'oris', freqOris, 'oriBW', biasOriBW, 'labels', labels);
end
