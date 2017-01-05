function dbnMM = dbnMMpretrainAndLoad(dbnMM, modality, X, opts)
% This function train the individual feature extraction part
% of the multimodal model.

assert(dbnMM.cvMultiModalArch{modality}(1) == size(X,2),'Error with data dimension');

dbn.sizes = dbnMM.cvMultiModalArch{modality}(2:end);
dbn = dbnsetup(dbn, X, opts);
dbnMM.dbn{modality} = dbntrain(dbn, X, opts);

end