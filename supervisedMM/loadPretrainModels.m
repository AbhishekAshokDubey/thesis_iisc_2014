function multiModalModel = loadPretrainModels(multiModalModel, pretarinedModel)
% function to initialize the weights of the network from a pretrained model

if pretarinedModel.modelType == 'dbn'
    for i = 1 : pretarinedModel.iNoOfModlaities
        for j = 1 : numel(pretarinedModel.cvMultiModalArch{1})-1
            nnMultiModal.W{j}{i} = [pretarinedModel.dbn{i}.rbm{j}.c pretarinedModel.dbn{i}.rbm{j}.W];
        end
    end
    lastLayerCopied = numel(pretarinedModel.cvMultiModalArch{1}) -1 ;
    for i = 1 : numel(pretarinedModel.vMainModelArch)
        nnMultiModal.W{i+lastLayerCopied} = [pretarinedModel.dbn{pretarinedModel.iNoOfModlaities+1}.rbm{i}.c pretarinedModel.dbn{pretarinedModel.iNoOfModlaities+1}.rbm{i}.W];
    end
    
elseif pretarinedModel.modelType == 'sae'
    for i = 1 : pretarinedModel.iNoOfModlaities
        for j = 1 : numel(pretarinedModel.cvMultiModalArch{1})-1
            nnMultiModal.W{j}{i} = pretarinedModel.sae{i}.ae{j}.W{1};
        end
    end
    lastLayerCopied = numel(pretarinedModel.cvMultiModalArch{1}) -1 ;
    for i = 1 : numel(pretarinedModel.vMainModelArch)
        nnMultiModal.W{i+lastLayerCopied} = pretarinedModel.sae{pretarinedModel.iNoOfModlaities+1}.ae{i}.W{1};
    end
    
elseif pretarinedModel.modelType == 'dae'
    preTrainedLayerCount = (pretarinedModel.iTotalNoOfLayers - 1)/2;
    for i = 1:preTrainedLayerCount
        nnMultiModal.W{i} = pretarinedModel.W{i};
    end
end
end