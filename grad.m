function [gradident] = grad(y,w,kernel,lambda)
% y= dataset y
% w= weight values
% kernel= kernel
% lambda = 

% Calculating the gradient using sigmoid function
n=size(y,1);
gradident=zeros(1000,1);
for i=1:n
gradident =gradident + (sigmoid(y(i) * w' * kernel(i,:)' ) -1 ) * y(i)*kernel(i,:)';
end
gradident=gradident/n +  2* lambda * w;