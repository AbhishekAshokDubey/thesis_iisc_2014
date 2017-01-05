function saeMM = saeMMtrain(saeMM, x, opts)
% Function to train the Multimodal SAE
% Note: SAE are just used for pretraing and not as generative models.

sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;

data = {};
cumSum = 0;
for i=1:saeMM.iNoOfModlaities
    data{i} = x(:,cumSum+1: cumSum+ saeMM.cvMultiModalArch{i}(1));
    cumSum = cumSum + saeMM.cvMultiModalArch{i}(1);
end

topLayerData = [];

for i=1:saeMM.iNoOfModlaities
    for j = 1 : numel(saeMM.sae{i}.ae)
        saeMM.sae{i}.ae{j}.activation_function = opts.activation_function;
        saeMM.sae{i}.ae{j}.learningRate = opts.learningRate;
        saeMM.sae{i}.ae{j}.inputZeroMaskedFraction = opts.inputZeroMaskedFraction;
        saeMM.sae{i}.ae{j}.momentum = opts.momentum;
    end
    saeMM.sae{i} = saetrain(saeMM.sae{i}, data{i}, opts);
    temp = [];
    
    nntemp = nnsetup(saeMM.cvMultiModalArch{i});
    for j = 1: numel(saeMM.cvMultiModalArch{i}) - 1
        nntemp.W{j} = saeMM.sae{i}.ae{j}.W{1};
    end
    
    nntemp.activation_function = opts.activation_function;
    nntemp = nnff(nntemp, data{i});
    temp = nntemp.a{nntemp.n};
    
    topLayerData = [topLayerData temp];
end

for j = 1 : numel(saeMM.sae{saeMM.iNoOfModlaities+ 1}.ae)
    saeMM.sae{saeMM.iNoOfModlaities+ 1}.ae{j}.activation_function = opts.activation_function;
    saeMM.sae{saeMM.iNoOfModlaities+ 1}.ae{j}.learningRate = opts.learningRate;
    saeMM.sae{saeMM.iNoOfModlaities+ 1}.ae{j}.inputZeroMaskedFraction = opts.inputZeroMaskedFraction;
    saeMM.sae{saeMM.iNoOfModlaities+ 1}.ae{j}.momentum = opts.momentum;
end
saeMM.sae{saeMM.iNoOfModlaities+ 1} = saetrain(saeMM.sae{saeMM.iNoOfModlaities + 1}, topLayerData, opts);
end