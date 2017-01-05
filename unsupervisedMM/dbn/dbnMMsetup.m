function dbnMM = dbnMMsetup(vMainModelArch, cvMultiModalArch, opts)

% This function Initializes a multimodal DBN.
% If we have n modalities then,
% a total of 'n+1' DBN are setup, one for each modality
% and one at the top of all these dbn.

dbnMM.iNoOfModlaities = numel(cvMultiModalArch);
dbnMM.vMainModelArch = vMainModelArch;
dbnMM.cvMultiModalArch = cvMultiModalArch;
dbnMM.modelType = 'dbn';

topLayerUnitCount = 0;

for j = 1:dbnMM.iNoOfModlaities
    dummyData = rand(2,cvMultiModalArch{j}(1));
    dbn.sizes = cvMultiModalArch{j}(2:end);
    dbn = dbnsetup(dbn, dummyData, opts);
    dbnMM.dbn{j} = dbn;
    
    topLayerUnitCount = topLayerUnitCount + cvMultiModalArch{j}(end);
end

dummyData = rand(2,topLayerUnitCount);
dbn.sizes = vMainModelArch;
dbn = dbnsetup(dbn, dummyData, opts);
dbnMM.dbn{dbnMM.iNoOfModlaities + 1} = dbn;
end