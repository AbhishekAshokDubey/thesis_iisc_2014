function nn = nnapplygradsMM(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases

    for i = 1: nn.iMultiModalLayerCount-1
        %i
        for j = 1:nn.iNoOfModlaities
            dW = nn.dW{i}{j};
            dW = nn.learningRate * dW;

            if(nn.momentum>0)
                nn.vW{i}{j} = nn.momentum * nn.vW{i}{j} + dW;
                dW = nn.vW{i}{j};
            end
            nn.W{i}{j} = nn.W{i}{j} - dW;
        end
    end

    iLastMainModellayerNo =  nn.iMultiModalLayerCount + nn.iMainModalLayerCount - 1;
    for i = nn.iMultiModalLayerCount : iLastMainModellayerNo - 1
        %i
        dW = nn.dW{i};
        dW = nn.learningRate * dW;

        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
        nn.W{i} = nn.W{i} - dW;
    end

    for i = iLastMainModellayerNo : nn.iTotalNoOfLayers-1
        %i
        for j = 1:nn.iNoOfModlaities
            dW = nn.dW{i}{j};
            dW = nn.learningRate * dW;

            if(nn.momentum>0)
                nn.vW{i}{j} = nn.momentum * nn.vW{i}{j} + dW;
                dW = nn.vW{i}{j};
            end
            nn.W{i}{j} = nn.W{i}{j} - dW;
        end
    end
%disp('Here it is done');
end