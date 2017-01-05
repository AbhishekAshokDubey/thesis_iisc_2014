function [daeMM,L] = daeMMtrain(daeMM, X, opts)
[daeMM,L] = nntrainMM(daeMM, X, X, opts);
end