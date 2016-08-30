function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%

  for example_idx = 1:size(X, 2)
      example = X(:,example_idx);
      label = y(example_idx);
      sigm = 1 / (1 + exp(-1 * theta' * example));
      f = f - (label * log(sigm) + (1 - label) * log(1 - sigm));
      
      for feature_idx = 1:size(g)
          g(feature_idx) = g(feature_idx) + example(feature_idx) * (sigm - label);
      end
  end