% Examples for using the Library

% Note for this entire library uses Two important variable/ structure for
% the specification of the multimodal architecture.
% 1) cvMultiModalArch : This is a cell array. A cell here (cvMultiModalArch{i}) represents the
%                       the architecture (number of hidden usnits in each layer)
%                       used for a (i th) modality.
% 2) vMainModelArch : This is a vectore which represent the architecture
%                     (number of hidden usnits in each layer) for the model
%                     on top of the multimodal structure.


% Data: we load 'MNISTtraining.mat' for the MNIST tarining dataset
% Images are stored in 'trainImg' variable and corresponding labels in
% 'trainLabel' variable.

clear;clc;
load('MNISTtraining.mat');
vMainModelArch = [100 50];          % architecture for the model on top of each Modality
cvMultiModalArch{1} = [784 70];     % architecture for first Modality
cvMultiModalArch{2} = [10 50];      % architecture for second Modality
X = [trainImg trainLabel];


%%%%%%%%%%%% unsupervisedMM/ DBN %%%%%%%%%%%%%%
opts.numepochs =   2;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;

dbnMM = dbnMMsetup(vMainModelArch, cvMultiModalArch, opts);

modality = 1;
% imshow(reshape(dbnMM.dbn{modality}.rbm{1}.W(1,:), 28, 28));
% figure; imshow(reshape(dbnMM.dbn{modality}.rbm{1}.W(2,:), 28, 28));
dbnMM = dbnMMpretrainAndLoad(dbnMM, modality, trainImg, opts);
% figure; imshow(reshape(dbnMM.dbn{modality}.rbm{1}.W(1,:), 28, 28));
% figure; imshow(reshape(dbnMM.dbn{modality}.rbm{1}.W(2,:), 28, 28));

dbnMM = dbnMMtrain(dbnMM, X, opts);

fillModality = 1;
sampleCount = 2;

Xincomp = [rand(size(trainImg)) trainLabel];
% Xincomp = [zeros(size(trainImg)) trainLabel];

generatedModality = dbnGenerateModality(dbnMM, Xincomp, fillModality, sampleCount);
% i = 10
%generatedModality = dbnGenerateModality(dbnMM, Xincomp(i,:), fillModality, sampleCount);
%Xincomp(i,end-9:end)
%imshow(reshape(generatedModality,28,28))

testModality = 1;
Xtest = [zeros(size(trainImg)) trainLabel];
Xmissing = trainImg;
sampleCount = 5;
bMultiLabel = true;
repeatCount = 1;
errorCount = dbnMMtest(dbnMM, Xtest, Xmissing, testModality, sampleCount, bMultiLabel, repeatCount);

testModality = 2;
Xtest = [trainImg zeros(size(trainLabel))];
Xmissing = trainLabel;
sampleCount = 5;
bMultiLabel = false;
repeatCount = 1;
errorCount = dbnMMtest(dbnMM, Xtest, Xmissing, testModality, sampleCount, bMultiLabel, repeatCount);




%%%%%%%%%%%% unsupervisedMM/ SAE %%%%%%%%%%%%%%
% Note: SAE are just used for pretraing and not as generative models.

saeMM = saeMMsetup(vMainModelArch, cvMultiModalArch);
% check: saeMM.sae{i}.ae{1}

opts.activation_function = 'sigm';
opts.learningRate              = 1;
opts.inputZeroMaskedFraction   = 0.5;
opts.momentum  =   0;
opts.numepochs =   1;
opts.batchsize = 100;

modality = 1;
% imshow(reshape(saeMM.sae{1}.ae{1}.W{1}(1,2:end), 28, 28));
% figure; imshow(reshape(saeMM.sae{1}.ae{1}.W{1}(2,2:end), 28, 28));
saeMM = saeMMpretrainAndLoad(saeMM, modality, trainImg, opts);
% figure; imshow(reshape(saeMM.sae{1}.ae{1}.W{1}(1,2:end), 28, 28));
% figure; imshow(reshape(saeMM.sae{1}.ae{1}.W{1}(2,2:end), 28, 28));

saeMM = saeMMtrain(saeMM, X, opts);




%%%%%%%%%%%% unsupervisedMM/ DAE %%%%%%%%%%%%%%
daeMM = daeMMsetup(vMainModelArch, cvMultiModalArch);

opts.numepochs =  1;
opts.batchsize = 100;
daeMM.activation_function = 'sigm';
daeMM.learningRate = 1;

modality = 1;
% imshow(reshape(daeMM.W{1}{modality}(1,2:end), 28, 28));
% figure; imshow(reshape(daeMM.W{1}{modality}(1,2:end), 28, 28));
daeMM = daeMMpretrainAndLoad(daeMM, modality, trainImg, opts);
% figure; imshow(reshape(daeMM.W{1}{modality}(1,2:end), 28, 28));
% figure; imshow(reshape(daeMM.W{1}{modality}(1,2:end), 28, 28));

opts.numepochs =  2;
daeMM = daeMMtrain(daeMM, X, opts);

fillModality = 1;
sampleCount = 2;

Xincomp = [rand(size(trainImg)) trainLabel];
% Xincomp = [zeros(size(trainImg)) trainLabel];

generatedModality = daeGenerateModality(daeMM, Xincomp, fillModality, sampleCount);
% i = 20
%generatedModality = daeGenerateModality(daeMM, Xincomp(i,:), fillModality, sampleCount);
%Xincomp(i,end-9:end)
%imshow(reshape(generatedModality,28,28))

testModality = 1;
Xtest = [zeros(size(trainImg)) trainLabel];
Xmissing = trainImg;
sampleCount = 5;
bMultiLabel = true;
repeatCount = 1;
errorCount = daeMMtest(daeMM, Xtest, Xmissing, testModality, sampleCount, bMultiLabel, repeatCount);

testModality = 2;
Xtest = [trainImg zeros(size(trainLabel))];
Xmissing = trainLabel;
sampleCount = 5;
bMultiLabel = false;
repeatCount = 1;
errorCount = daeMMtest(daeMM, Xtest, Xmissing, testModality, sampleCount, bMultiLabel, repeatCount);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear;clc;
load('MNISTtraining.mat');
vMainModelArch = [100 10];          % architecture for the model on top of each Modality
cvMultiModalArch{1} = [784 70];     % architecture for first Modality
cvMultiModalArch{2} = [10 50];      % architecture for second Modality
X = [trainImg trainLabel];
labels = rand(size(X,1), vMainModelArch(end)); % Random labels created to show the use of the function.

%%%%%%%%%%%% supervisedMM/ MMNN %%%%%%%%%%%%%%


% Ex. 1: No pretraning
nnMM = nnsetupMM(vMainModelArch, cvMultiModalArch);



opts.batchsize = 1000;
opts.numepochs = 2;
nnMM.activation_function = 'sigm';
nnMM = nntrainMM(nnMM, X, labels, opts);

bMultiLabel = true;
errorCount = nntestMM(nnMM, X, labels, bMultiLabel);




% Ex. 2: DBN pretraining
opts.numepochs =   2;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbnMM = dbnMMsetup(vMainModelArch, cvMultiModalArch, opts);

modality = 1;
% Both the lines below are optional, any (or both) of these can be used for
% better initialization of the network
dbnMM = dbnMMpretrainAndLoad(dbnMM, modality, trainImg, opts);
dbnMM = dbnMMtrain(dbnMM, X, opts);

nnMM = nnsetupMM(vMainModelArch, cvMultiModalArch);
nnMM = loadPretrainModels(nnMM, dbnMM);

opts.batchsize = 150;
opts.numepochs = 2;
nnMM.activation_function = 'sigm';
nnMM = nntrainMM(nnMM, X, labels, opts);

bMultiLabel = true;
errorCount = nntestMM(nnMM, X, labels, bMultiLabel);




% Ex. 3: SAE pretraining
saeMM = saeMMsetup(vMainModelArch, cvMultiModalArch);
% check: saeMM.sae{i}.ae{1}

opts.activation_function = 'sigm';
opts.learningRate              = 1;
opts.inputZeroMaskedFraction   = 0.5;
opts.momentum  =   0;
opts.numepochs =   1;
opts.batchsize = 100;

modality = 1;
% Both the lines below are optional, any (or both) of these can be used for
% better initialization of the network
saeMM = saeMMpretrainAndLoad(saeMM, modality, trainImg, opts);
saeMM = saeMMtrain(saeMM, X, opts);

nnMM = nnsetupMM(vMainModelArch, cvMultiModalArch);
nnMM = loadPretrainModels(nnMM, saeMM);

opts.batchsize = 150;
opts.numepochs = 2;
nnMM.activation_function = 'sigm';
nnMM = nntrainMM(nnMM, X, labels, opts);

bMultiLabel = true;
errorCount = nntestMM(nnMM, X, labels, bMultiLabel);




% Ex. 4: DAE pretraining
daeMM = daeMMsetup(vMainModelArch, cvMultiModalArch);

opts.numepochs =  1;
opts.batchsize = 100;
daeMM.activation_function = 'sigm';

modality = 1;
% Both the lines below are optional, any (or both) of these can be used for
% better initialization of the network
daeMM = daeMMpretrainAndLoad(daeMM, modality, trainImg, opts);
daeMM = daeMMtrain(daeMM, X, opts);

nnMM = nnsetupMM(vMainModelArch, cvMultiModalArch);
nnMM = loadPretrainModels(nnMM, daeMM);

opts.batchsize = 1000;
opts.numepochs = 2;
nnMM.activation_function = 'sigm';
nnMM = nntrainMM(nnMM, X, labels, opts);

bMultiLabel = true;
errorCount = nntestMM(nnMM, X, labels, bMultiLabel);