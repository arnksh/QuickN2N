function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
m=size(data,2);


exp_pow = theta*data;
exp_pow = bsxfun(@minus, exp_pow, max(exp_pow, [], 1));
total = sum(exp(exp_pow),1);
hypothesis = exp(exp_pow)./repmat(total,size(exp_pow,1),1);


each_k = groundTruth.*log(hypothesis); 
regularization = lambda/2 * sum(sum(theta.^2));
simple_cost = -1/m*sum(sum(each_k,1));
cost = simple_cost+regularization;



difference = groundTruth - hypothesis;  
simple_thetagrad = -1/m*(difference * data');
regularization_grad = lambda * theta; 
thetagrad = simple_thetagrad + regularization_grad;






% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

