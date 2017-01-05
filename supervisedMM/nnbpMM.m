function nn = nnbpMM(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights

% Remember for backpropogation: Given the gradient wrt to 'g' a parameter/ layer
% to compute the gradient of error with
% respect to any parameter/ layer 'f',
% Given dE/dg, to find dE/df: (compute and use: dg/df)
% dE/df = dg/df * dE/dg

n = nn.iTotalNoOfLayers;
switch nn.output
    % error gradient (for the output layer) wrt to the inputs of the
    % output layer
    case 'sigm'
        d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
    case {'softmax','linear'}
        d{n} = - nn.e;
end

if nn.bPretraining
    temp = {};
    cumSum=0;
    for j=1:nn.iNoOfModlaities
        temp{j} = d{n}(:,cumSum+1: cumSum+nn.cvMultiModalArch{j}(1));
        cumSum = cumSum + nn.cvMultiModalArch{j}(1);
    end
    d{n} = temp;
    
    for i = n-1: -1: n - nn.iMultiModalLayerCount + 1
        switch nn.activation_function
            case 'sigm'
                for j = 1:nn.iNoOfModlaities
                    d_act{j} = nn.a{i}{j} .* (1 - nn.a{i}{j});
                end
            case 'tanh_opt'
                for j = 1:nn.iNoOfModlaities
                    d_act{j} = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}{j}.^2);
                end
        end
        % Backpropagate first derivatives
        if i+1==n
            for j = 1:nn.iNoOfModlaities
                d{i}{j} = (d{i + 1}{j} * nn.W{i}{j}) .* d_act{j};
            end
        else
            for j = 1:nn.iNoOfModlaities
                d{i}{j} = (d{i + 1}{j}(:,2:end) * nn.W{i}{j}) .* d_act{j};
            end
        end
    end
    
    % combining the output of all modalities in joint layer
    %m = size(d{n}{1},1);
    %temp = [ones(m,1)];
    temp = [];
    for j = 1:nn.iNoOfModlaities
        temp = [temp d{i}{j}(:,2:end)];
    end
    d{i} = temp;
end

mainModalEndInx = 0;
mainModalStartInx = 0;
if nn.bPretraining
    mainModalEndInx = n - nn.iMultiModalLayerCount;
else
    mainModalEndInx = n-1;
end
mainModalStartInx = nn.iMultiModalLayerCount;

% mainmodel (middle) backpropogation
for i = mainModalEndInx : -1 : mainModalStartInx
    switch nn.activation_function
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i});
        case 'tanh_opt'
            d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
    end
    if i == mainModalEndInx
        d{i} = (d{i + 1}(:,1:end) * nn.W{i}) .* d_act;
    else
        d{i} = (d{i + 1}(:,2:end) * nn.W{i}) .* d_act;
    end
end

temp = {};
cumSum=0;
d{mainModalStartInx} = d{mainModalStartInx}(:,2:end);
for j=1:nn.iNoOfModlaities
    temp{j} = d{mainModalStartInx}(:,cumSum+1: cumSum+nn.cvMultiModalArch{j}(end));
    cumSum = cumSum + nn.cvMultiModalArch{j}(end);
end
d{mainModalStartInx} = temp;

d_act = {};
% bottom layer multimodal backpropogation
for i = (nn.iMultiModalLayerCount-1):-1:2
    switch nn.activation_function
        case 'sigm'
            for j = 1:nn.iNoOfModlaities
                d_act{j} = nn.a{i}{j} .* (1 - nn.a{i}{j});
            end
        case 'tanh_opt'
            for j = 1:nn.iNoOfModlaities
                d_act{j} = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * (nn.a{i}{j}).^2);
            end
    end
    
    % Backpropagate first derivatives
    if i == nn.iMultiModalLayerCount-1
        for j = 1:nn.iNoOfModlaities
            d{i}{j} = (d{i + 1}{j} * nn.W{i}{j}) .* d_act{j}; % Bishop (5.56)
        end
    else % in this case in d{i} the bias term has to be removed
        for j = 1:nn.iNoOfModlaities
            d{i}{j} = (d{i + 1}{j}(:,2:end) * nn.W{i}{j}) .* d_act{j};
        end
    end
end

% d{1}
% d{2}
% size(d{3}{1})
% size(d{3}{2})
% size(d{3}{3})
% size(d{4})
% size(d{5})
% size(d{6})
% size(d{7})
% d{8}
% d{9}

% nn.a{1}
% nn.a{2}
% size(nn.a{3})
% size(nn.a{4})
% size(nn.a{5})
% size(nn.a{6})
% size(nn.a{7}{1})
% size(nn.a{7}{2})
% size(nn.a{7}{3})
% nn.a{8}
% size(nn.a{9})

for i = 1: nn.iMultiModalLayerCount-1
    if i == nn.iMultiModalLayerCount-1
        for j = 1:nn.iNoOfModlaities
            nn.dW{i}{j} = (d{i + 1}{j}' * nn.a{i}{j}) / size(d{i + 1}{j}, 1);
        end
    else
        for j = 1:nn.iNoOfModlaities
            nn.dW{i}{j} = (d{i + 1}{j}(:,2:end)' * nn.a{i}{j}) / size(d{i + 1}{j}, 1);
        end
    end
end

iLastMainModellayerNo =  nn.iMultiModalLayerCount + nn.iMainModalLayerCount - 1;
for i = nn.iMultiModalLayerCount : iLastMainModellayerNo - 1
    if i == (iLastMainModellayerNo - 1)
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
    else
        nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);
    end    
end

for i = iLastMainModellayerNo : n-1
    if i == n-1
        for j = 1:nn.iNoOfModlaities
            nn.dW{i}{j} = (d{i + 1}{j}' * nn.a{i}{j}) / size(d{i + 1}{j}, 1);
        end
    else
        for j = 1:nn.iNoOfModlaities
            nn.dW{i}{j} = (d{i + 1}{j}(:,2:end)' * nn.a{i}{j}) / size(d{i + 1}{j}, 1);
        end
    end
end
end