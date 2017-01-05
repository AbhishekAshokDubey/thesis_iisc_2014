function errorCount = nntestMM(nnMM, X, label, bMultiLabel)

nnMM = nnffMM(nnMM, X);

if bMultiLabel
    errorCount = nnMM.L;
else
    [temp indexPredicted] = max(Ypredicted,[],2);
    [temp index] = max(label,[],2);
    error = (indexPredicted ~= index);
    errorCount = sum(error);
end

end