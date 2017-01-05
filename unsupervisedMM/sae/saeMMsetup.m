function saeMM = saeMMsetup(vMainModelArch, cvMultiModalArch)
% Initializes the multimodal SAE for pretraining.
% Note: SAE are just used for pretraing and not as generative models.
% For 'n' modal data, first n SAE are for each modality and 'n+1'th SAE is
% on the top of these modalities

saeMM.iNoOfModlaities = numel(cvMultiModalArch);
saeMM.vMainModelArch = vMainModelArch;
saeMM.cvMultiModalArch = cvMultiModalArch;
saeMM.modelType = 'sae';

topLayerUnitCount = 0;

for i = 1:saeMM.iNoOfModlaities
    sae = saesetup(cvMultiModalArch{i});
    saeMM.sae{i} = sae;
    
    topLayerUnitCount = topLayerUnitCount + cvMultiModalArch{i}(end);
end

sae = saesetup([topLayerUnitCount vMainModelArch]);
saeMM.sae{saeMM.iNoOfModlaities+1} = sae;
end