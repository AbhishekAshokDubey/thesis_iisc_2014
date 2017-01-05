clear; clc;
vMainModelArch = [100 50];
cvMultiModalArch{1} = [700 70 5];
cvMultiModalArch{2} = [100 70 5];
cvMultiModalArch{3} = [200 70 5];
bPretraining = true;

nn = nnsetupMM(vMainModelArch, cvMultiModalArch, bPretraining);

X = rand(300,1000);
Y = rand(300,10);

opts.batchsize = 150;
opts.numepochs = 2;

nn = nntrainMM(nn, X, X, opts);