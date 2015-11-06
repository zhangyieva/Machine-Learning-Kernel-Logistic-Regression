function cost = CostFunction(y,w,kernel,lambda)
% y= trainning y dataset 
% w= weight values
% kernel= training kernel
% lambda = lambda

% Calculating Logistc loss
n=size(y,1);
cost=0;
for i=1:n
cost =cost + log(sigmoid(y(i) * w' * kernel(i,:)' )) ;
end
cost= -cost/n + lambda * (w' * w);