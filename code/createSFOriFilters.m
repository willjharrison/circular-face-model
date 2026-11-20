function [allFilts, filtSteps, allOris] = createSFOriFilters(imFFTSize, nFreqs, nOris, minFreq, maxFreq, oriBW)
% Creates a bank of 2D spatial frequency and orientation filters in FFT space
% forming a seamless partition of unity across the analytic half-plane.

if nargin < 6 || isempty(oriBW)
    oriBW = 180 / nOris;
end

[angDist, radDist] = polarDistFun(imFFTSize);

% Expand bounds by one log-step so interior bandpass filters sum to 1 across [minFreq, maxFreq]
if nFreqs > 1
    logStep   = (log10(maxFreq) - log10(minFreq)) / (nFreqs - 1);
    lowBound  = 10^(log10(minFreq) - logStep);
    highBound = 10^(log10(maxFreq) + logStep);
else
    lowBound  = minFreq / 2;
    highBound = maxFreq * 2;
end

[sfFilts, filtSteps] = createBandPassFilters_customSizes(radDist, nFreqs + 2, lowBound, highBound);
sfFilts   = sfFilts(:, :, 2:end-1);
filtSteps = filtSteps(2:end-1);

allOris = linspace(0, 180, nOris + 1);
allOris(end) = [];

% Analytic right half-plane mask (each conjugate pair counted once)
inHalfPlane = (angDist >= -90) & (angDist < 90);

[nRows, nCols] = size(radDist);
allFilts = zeros(nRows, nCols, nOris, nFreqs);

for sfLoop = 1:nFreqs
    sfSlice = sfFilts(:, :, sfLoop) .* inHalfPlane;
    for oriLoop = 1:nOris
        thisOri = allOris(oriLoop) - 90;
        % Periodic angular distance in [-90, 90) so -90 deg wraps seamlessly to +90 deg
        dAng    = mod(angDist - thisOri + 90, 180) - 90;
        oriFilt = 0.5 * (1 + cos(pi * min(abs(dAng) / oriBW, 1)));
        allFilts(:, :, oriLoop, sfLoop) = fftshift(sfSlice .* oriFilt);
    end
end
end
