function d = daeGenerateModality(daeMM, X, fillModality, sampleCount)

cumSum = 0;
fillModalityStartIndx = 0;
fillModalityEndIndx = 0;

for i = 1 : daeMM.iNoOfModlaities
    if i == fillModality
        fillModalityStartIndx = cumSum+1; 
        fillModalityEndIndx = cumSum+ daeMM.cvMultiModalArch{i}(1);
    end        
    cumSum = cumSum + daeMM.cvMultiModalArch{i}(1);
end

for i = 1 : sampleCount
daeMM = nnffMM(daeMM, X);
X(:,fillModalityStartIndx:fillModalityEndIndx) = daeMM.a{daeMM.iTotalNoOfLayers}(:,fillModalityStartIndx:fillModalityEndIndx);
end

d = X(:,fillModalityStartIndx:fillModalityEndIndx);
end