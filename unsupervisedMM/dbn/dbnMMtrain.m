function dbnMM = dbnMMtrain(dbnMM, X, opts)
% This function layer wise train the multimodal DBN

    data = {};
    cumSum = 0;

    for i=1:dbnMM.iNoOfModlaities
        data{i} = X(:,cumSum+1: cumSum+ dbnMM.cvMultiModalArch{i}(1));
        cumSum = cumSum + dbnMM.cvMultiModalArch{i}(1);
    end

    topLayerData = [];

    for i=1:dbnMM.iNoOfModlaities
        dbnMM.dbn{i} = dbntrain(dbnMM.dbn{i}, data{i}, opts);
        temp = data{i};
        for j=1:numel(dbnMM.dbn{i}.rbm)
            temp = rbmup(dbnMM.dbn{i}.rbm{j},temp);
        end
        topLayerData = [topLayerData temp];
    end

    dbnMM.dbn{dbnMM.iNoOfModlaities+ 1} = dbntrain(dbnMM.dbn{dbnMM.iNoOfModlaities + 1}, topLayerData, opts);
end