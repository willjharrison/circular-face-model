function [sfFilts,filtSteps] = createBandPassFilters_customSizes(radDist,nFreqs,lowPassFreq,highPassFreq)

filtSteps = logspace(log10(lowPassFreq),log10(highPassFreq),nFreqs);

lpFilter = createLowPassFilter(log(radDist),log(filtSteps(1)),log(filtSteps(2)/filtSteps(1)));
hpFilter = 1 - createLowPassFilter(log(radDist),log(filtSteps(end-1)),log(filtSteps(end)/filtSteps(end-1)));

bandFilt = zeros(length(radDist),length(radDist),nFreqs-2);
for filtLoop = 1:nFreqs-2
    
    bandFilt(:,:,filtLoop) = raisedCos(radDist,filtSteps(filtLoop+1),filtSteps(filtLoop),'log');
    
end

sfFilts = cat(3,lpFilter,bandFilt,hpFilter);
