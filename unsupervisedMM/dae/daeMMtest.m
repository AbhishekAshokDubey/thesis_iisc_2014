function errorCount = daeMMtest(daeMM, X, label, fillModality, sampleCount, bMultiLabel, repeatCount)

d = daeGenerateModality(daeMM, X, fillModality, sampleCount);

actualColumnCount = floor(size(d,2)/repeatCount);

Ypredicted = reshape(d,size(d,1), actualColumnCount, repeatCount);
Ypredicted = mean(Ypredicted,3);

if bMultiLabel
    errorCount = sum(sum((Ypredicted - label).^2));
else
    [temp indexPredicted] = max(Ypredicted,[],2);
    [temp index] = max(label,[],2);
    error = (indexPredicted ~= index);
    errorCount = sum(error);
end
end