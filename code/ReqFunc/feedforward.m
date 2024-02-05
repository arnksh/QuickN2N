function [y,y_dot] = feedforward(x,W,b,l)
    y = x;
    y_dot = cell(1,l-1);

    for i = 1:l-1
        y{i+1} = sigmoid(W{i}*y{i}+b{i});
        y_dot{i} = sigmoidPrime(W{i}*y{i}+b{i});
    end
end


function y = sigmoid(X)
%SIGMOID    computes the value of the neutron using the sigmoid function
%   y = SIGMOID(X) returns the sigmoid function of the elements of vector
%   X, i.e. y = 1./(1+exp(-X))
%
%   see also: exp

y = 1./(1+exp(-X));
end

