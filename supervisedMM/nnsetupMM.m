function nn = nnsetupMM(vMainModelArch, cvMultiModalArch, bPretraining)
% This function setup a multimodal neural network.
% When 'bPretraining' is false it acts as supervised multimodal neural
% network. When 'bPretraining' is true it acts as deep multimodal auto
% encoder.


if ~exist('bPretraining','var')
    bPretraining = false;
end

iNoOfModlaities = numel(cvMultiModalArch);
iMultiModalLayerCount = numel(cvMultiModalArch{1});

combinedFeatureSize = 0;
for i=1:iNoOfModlaities
    combinedFeatureSize = combinedFeatureSize + cvMultiModalArch{i}(end);
end

vMainModelArch = [combinedFeatureSize vMainModelArch];
if bPretraining
    vMainModelArch = [vMainModelArch(1:end-1) fliplr(vMainModelArch)];
end
iMainModalLayerCount = numel(vMainModelArch);

if bPretraining
    iTotalNoOfLayers = 2*(iMultiModalLayerCount - 1) + iMainModalLayerCount;
else
    iTotalNoOfLayers = iMultiModalLayerCount + iMainModalLayerCount - 1;
end


for i = 1:iMultiModalLayerCount-1
    nn.p{i+1} = [];
    if bPretraining
        nn.p{iTotalNoOfLayers - i + 1} = [];
    end
    for j = 1:iNoOfModlaities
        nn.W{i}{j} = (rand(cvMultiModalArch{j}(i+1), cvMultiModalArch{j}(i)+1) - 0.5) * 2 * 4 * sqrt(6 / (cvMultiModalArch{j}(i+1) + cvMultiModalArch{j}(i)));
        nn.vW{i}{j} = zeros(size(nn.W{i}{j}));
        nn.p{i+1} = [nn.p{i+1} zeros(1,cvMultiModalArch{j}(i+1))]; 
        if bPretraining
            nn.W{iTotalNoOfLayers - i}{j} = (rand(cvMultiModalArch{j}(i), cvMultiModalArch{j}(i+1)+1) - 0.5) * 2 * 4 * sqrt(6 / (cvMultiModalArch{j}(i+1) + cvMultiModalArch{j}(i)));
            nn.vW{iTotalNoOfLayers - i}{j} = zeros(size(nn.W{iTotalNoOfLayers - i}{j}));
            nn.p{iTotalNoOfLayers - i + 1} = [nn.p{iTotalNoOfLayers - i + 1} zeros(1,cvMultiModalArch{j}(i))];
        end
    end
%     nn.splitWeights(i) = true;
%     nn.splitWeights(iTotalNoOfLayers - i) = true;
end

for i = 1:iMainModalLayerCount - 1
    nn.W{iMultiModalLayerCount - 1 + i} = (rand(vMainModelArch(i+1), vMainModelArch(i)+1) - 0.5) * 2 * 4 * sqrt(6 / (vMainModelArch(i+1) + vMainModelArch(i)));
    nn.vW{iMultiModalLayerCount - 1 + i} = zeros(size(nn.W{iMultiModalLayerCount - 1 + i}));
%     nn.splitWeights(iMultiModalLayerCount - 1 + i) = false;
    
    nn.p{iMultiModalLayerCount + i} = zeros(1, vMainModelArch(i+1));
end

nn.bPretraining = bPretraining;
nn.iTotalNoOfLayers = iTotalNoOfLayers;
nn.vMainModelArch = vMainModelArch;
nn.cvMultiModalArch = cvMultiModalArch;
nn.iNoOfModlaities = iNoOfModlaities;
nn.iMultiModalLayerCount = iMultiModalLayerCount;
nn.iMainModalLayerCount = iMainModalLayerCount;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate                     = 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
nn.momentum                         = 0.5;          %  Momentum
nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'

end