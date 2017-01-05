function saeMM = saeMMpretrainAndLoad(saeMM, modality, X, opts)
% Function to pretrain the feature extraction part of a modality

dim = saeMM.cvMultiModalArch{modality};
sae = saesetup(dim);

for j = 1 : numel(sae.ae)
    sae.ae{j}.activation_function = opts.activation_function;
    sae.ae{j}.learningRate = opts.learningRate;
    sae.ae{j}.inputZeroMaskedFraction = opts.inputZeroMaskedFraction;
    sae.ae{j}.momentum = opts.momentum;
end
sae = saetrain(sae, X, opts);

saeMM.sae{modality} = sae;
end

