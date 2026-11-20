function normIm = normaliseImage(im, gamma, mask)
% NORMALISEIMAGE Gamma-corrects, subtracts mean, divides by standard
% deviation (RMS contrast normalisation), and applies the aperture mask.

if nargin < 2 || isempty(gamma)
    gamma = 2;
end
if nargin < 3 || isempty(mask)
    mask = ones(size(im));
end

thisIm = double(im);
thisIm(thisIm < 0) = 0;
thisIm(thisIm > 1) = 1;

thisIm = thisIm .^ (1 / gamma);
thisIm = thisIm - mean(thisIm(:));
sd = std(thisIm(:));
if sd > eps
    thisIm = thisIm / sd;
end

normIm = thisIm .* mask;
end
