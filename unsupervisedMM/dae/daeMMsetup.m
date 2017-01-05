function daeMM = daeMMsetup(vMainModelArch, cvMultiModalArch)

bPretraining = true;
daeMM = nnsetupMM(vMainModelArch, cvMultiModalArch, bPretraining);
daeMM.modelType = 'dae';
end