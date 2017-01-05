function [nn, L]  = nntrainMM(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net

% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');

% nargin: inbuilt matlab variable for number of arguments to the function
assert(nargin == 4 || nargin == 6,'number of input arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

% L stores the loss for each batch trained in each epoch
L = zeros(numepochs*numbatches,1);

n = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
%         %Add noise to input (for use in denoising autoencoder)
%         if(nn.inputZeroMaskedFraction ~= 0)
%             batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
%         end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnffMM(nn, batch_x, batch_y); 
        nn = nnbpMM(nn);
        nn = nnapplygradsMM(nn);
        
        L(n) = nn.L;        
        n = n + 1;
    end
    
%     nn.learningRate
%        if i == 49
%            disp('in');
%            nn.learningRate = 0.005;
%        end
    
    t = toc;

    nn = nnffMM(nn, train_x, train_y);
    str_perf = sprintf('; Full-batch train err = %f', nn.L);
            
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end
end