function errorCount = dbnMMtest(dbnMM, X, label, testModality, sampleCount, bMultiLabel, repeatCount)
% For testing of the multimodal model

d = dbnGenerateModality(dbnMM, X, testModality, sampleCount);

actualColumnCount = floor(size(d,2)/repeatCount);

Ypredicted = reshape(d,size(d,1), actualColumnCount, repeatCount);
Ypredicted = mean(Ypredicted,3);

if bMultiLabel
    errorCount = sum(sum((Ypredicted - label).^2)) / size(X,1);
else
    [temp indexPredicted] = max(Ypredicted,[],2);
    [temp index] = max(label,[],2);
    error = (indexPredicted ~= index);
    errorCount = sum(error);
end
end