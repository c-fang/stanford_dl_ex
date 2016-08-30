function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
%%% YOUR CODE HERE %%%

  for example_idx = 1:size(X, 2)
      example = X(:,example_idx);
      error = theta' * example - y(example_idx);
      f = f + 0.5 * (error) ^ 2;
      
      for feature_idx = 1:size(g)
          g(feature_idx) = g(feature_idx) + example(feature_idx) * error;
      end
  end
