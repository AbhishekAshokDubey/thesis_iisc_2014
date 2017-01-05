function daeMM = daeMMpretrainAndLoad(daeMM, modality, X, opts)

dim = daeMM.cvMultiModalArch{modality};
nn = nnsetup([dim fliplr(dim(1:end-1))]);
nn.activation_function = daeMM.activation_function;
nn = nntrain(nn, X, X, opts);

for i=1:numel(dim)-1
    daeMM.W{i}{modality} = nn.W{i};
end

end