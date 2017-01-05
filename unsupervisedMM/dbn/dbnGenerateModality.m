function d = dbnGenerateModality(dbnMM, X, fillModality, sampleCount)
% This function generates the missing modality from a multimodal
% architecture

data = {};
cumSum = 0;
for i = 1 : dbnMM.iNoOfModlaities
    data{i} = X(:,cumSum+1: cumSum+ dbnMM.cvMultiModalArch{i}(1));
    cumSum = cumSum + dbnMM.cvMultiModalArch{i}(1);
end

for i = 1 : sampleCount
    topLayerData = [];
    for j = 1 : dbnMM.iNoOfModlaities
        temp = data{j};
        for l = 1: numel(dbnMM.cvMultiModalArch{j}) - 1;
            temp = rbmup(dbnMM.dbn{j}.rbm{l}, temp);
        end
        topLayerData = [topLayerData temp];
    end
    temp = topLayerData;

    for l = 1 : numel(dbnMM.vMainModelArch)
        temp = rbmup(dbnMM.dbn{dbnMM.iNoOfModlaities+1}.rbm{l}, temp);
    end
    
    for l = numel(dbnMM.vMainModelArch) : -1 : 1
        temp = rbmdown(dbnMM.dbn{dbnMM.iNoOfModlaities+1}.rbm{l}, temp);
    end
    
    cumSum = 0;
    combinedData = temp;

    for j = 1 : dbnMM.iNoOfModlaities
        temp = combinedData(:,cumSum+1: cumSum+dbnMM.cvMultiModalArch{j}(end));
        cumSum = cumSum + dbnMM.cvMultiModalArch{j}(end);
        
        for l = numel(dbnMM.cvMultiModalArch{j}) - 1: -1: 1
            temp = rbmdown(dbnMM.dbn{j}.rbm{l}, temp);
        end
        if j == fillModality
            data{j} = temp;
        end
    end
end
d = data{fillModality};
end