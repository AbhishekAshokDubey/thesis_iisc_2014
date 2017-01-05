function nn = nnffMM(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

% nn.iMultiModalLayerCount 
% nn.iMainModalLayerCount 
% nn.iTotalNoOfLayers

n = nn.iTotalNoOfLayers;
m = size(x, 1);
i = 0;
cumSum = 0;
nn.a = {};
for j=1:nn.iNoOfModlaities
    nn.a{1}{j} = x(:,cumSum+1: cumSum+nn.cvMultiModalArch{j}(1));
    nn.a{1}{j} = [ones(m,1) nn.a{1}{j}];
    cumSum = cumSum + nn.cvMultiModalArch{j}(1);
end

%feedforward pass through multimodal model at bottom of the network
%disp('multiModal-Below');
for i = 2:nn.iMultiModalLayerCount
%    i
    switch nn.activation_function
        case 'sigm'
            for j = 1:nn.iNoOfModlaities
                nn.a{i}{j} = sigm(nn.a{i - 1}{j} * nn.W{i - 1}{j}');
                nn.a{i}{j} = [ones(m,1) nn.a{i}{j}];
            end
        case 'tanh_opt'
            for j = 1:nn.iNoOfModlaities
                nn.a{i}{j} = tanh_opt(nn.a{i - 1}{j} * nn.W{i - 1}{j}');
                nn.a{i}{j} = [ones(m,1) nn.a{i}{j}];
            end
    end
end

% combining the output of all modalities in joint layer
temp = [ones(m,1)];
for j = 1:nn.iNoOfModlaities
    temp = [temp nn.a{nn.iMultiModalLayerCount}{j}(:,2:end)];
end
nn.a{nn.iMultiModalLayerCount} = temp;

% feedforward pass for middle (main model) neural network except the last
% layer in the main model
%disp('MainModel below');
iLastMainModelLayer = nn.iMultiModalLayerCount + nn.iMainModalLayerCount - 1;
for i = nn.iMultiModalLayerCount+1 : iLastMainModelLayer-1
%    i
    switch nn.activation_function
        case 'sigm'
            nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
        case 'tanh_opt'
            nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
    end
    nn.a{i} = [ones(m,1) nn.a{i}];
end

% Case when we are not pretraining and so the last layer of the main model
% is the ouput layer
if ~nn.bPretraining
    assert(i + 1 == iLastMainModelLayer, 'error with the model specification');
    switch nn.output
        case 'sigm'
            nn.a{iLastMainModelLayer} = sigm(nn.a{iLastMainModelLayer - 1} * nn.W{iLastMainModelLayer - 1}');
        case 'linear'
            nn.a{iLastMainModelLayer} = nn.a{iLastMainModelLayer - 1} * nn.W{iLastMainModelLayer - 1}';
        case 'softmax'
            nn.a{iLastMainModelLayer} = nn.a{iLastMainModelLayer - 1} * nn.W{iLastMainModelLayer - 1}';
            nn.a{iLastMainModelLayer} = softmax(nn.a{iLastMainModelLayer});
    end
else
    %disp('Last Row main Model below');
    %iLastMainModelLayer
    switch nn.activation_function
        case 'sigm'
            nn.a{iLastMainModelLayer} = sigm(nn.a{iLastMainModelLayer - 1} * nn.W{iLastMainModelLayer - 1}');
        case 'tanh_opt'
            nn.a{iLastMainModelLayer} = tanh_opt(nn.a{iLastMainModelLayer - 1} * nn.W{iLastMainModelLayer - 1}');
    end
    
    temp = {};
    cumSum = 0;
    for j = 1:nn.iNoOfModlaities
        temp{j} = [ones(m,1) nn.a{iLastMainModelLayer}(:,cumSum+1: cumSum+nn.cvMultiModalArch{j}(end))];
        cumSum = cumSum + nn.cvMultiModalArch{j}(end);
    end
    nn.a{iLastMainModelLayer} = temp;
    
    iLastLayer = iLastMainModelLayer + nn.iMultiModalLayerCount- 1;
    % For top multimodal layers except last
    %disp('multiModal top layer Below');
    for i = iLastMainModelLayer+1 : iLastLayer - 1
        %i
        switch nn.activation_function
            case 'sigm'
                for j = 1:nn.iNoOfModlaities
                    nn.a{i}{j} = sigm(nn.a{i - 1}{j} * nn.W{i - 1}{j}');
                    nn.a{i}{j} = [ones(m,1) nn.a{i}{j}];
                end
            case 'tanh_opt'
                for j = 1:nn.iNoOfModlaities
                    nn.a{i}{j} = tanh_opt(nn.a{i - 1}{j} * nn.W{i - 1}{j}');
                    nn.a{i}{j} = [ones(m,1) nn.a{i}{j}];
                end
        end
    end
    
    assert(iLastLayer == n ,'Error with model initialization');
    %disp('last');
    %iLastLayer
    switch nn.output
        case 'sigm'
            for j =1:nn.iNoOfModlaities
                nn.a{iLastLayer}{j} = sigm(nn.a{iLastLayer - 1}{j}  * nn.W{iLastLayer - 1}{j}' );
            end
        case 'linear'
            for j = 1:nn.iNoOfModlaities
                nn.a{iLastLayer}{j} = nn.a{iLastLayer - 1}{j} * nn.W{iLastLayer - 1}{j}';
            end
        case 'softmax'
            for j =1:nn.iNoOfModlaities
                nn.a{iLastLayer}{j} = nn.a{iLastLayer - 1}{j} * nn.W{iLastLayer - 1}{j}';
                nn.a{iLastLayer}{j} = softmax(nn.a{iLastLayer}{j});
            end
            % If you dun have Neural Network toolbox, use following 2 lines of
            % code as substitute
            % nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            % nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));
    end
    temp = [];
    for j = 1:nn.iNoOfModlaities
        temp = [temp nn.a{iLastLayer}{j}];
    end
    nn.a{iLastLayer} = temp;
end

if exist('y', 'var')  % added by Abhishek
    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m;
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end % added by Abhishek

end