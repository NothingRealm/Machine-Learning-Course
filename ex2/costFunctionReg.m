function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[J, grad] = costFunction(theta, X, y);
sum_theta = theta' * theta;
sum_theta = sum_theta - theta(1) ^2;
norm = lambda / m;
sum_theta =  norm * sum_theta  / 2;
J = J + sum_theta;
grad = grad + theta * norm;
grad(1) = grad(1) - theta(1) * norm;
% =============================================================

end
