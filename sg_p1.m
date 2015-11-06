

% Stochastic Gradient Descent p=100
% 1. Test accuracy = 90.1000
% 2. Chosen step size = 0.01
% 3. Number of training iterations (until convergence or time limit)=710606
% 4. Total training time (in seconds)= total_time (which indicate the time
% takes for the gradient descent to converge)

n=length(TrainingX);

%criterien 
p=1;
stepsize=0.01;
lambda=0.01;
epsilon=0.01;


%initial values 

itr=0;
test_error_sg=0;
result_p1 = zeros(200,2);

%ramdom picking values 
m=randperm(1000,p);
X=TrainingX(m,:);
Y=TrainingY(m,:);
kernel=training_kernel(m,:);

%calculating new w with 0s 
ini_ws=zeros(n,1);
new_ws=ini_ws - stepsize* grad(Y,ini_ws,kernel,lambda);


%start time 
tic
timeval=tic;
time=0;
%start loop 
while time < total_time
    m=randperm(1000,p);
    X=TrainingX(m,:);
    Y=TrainingY(m,:);
    kernel = training_kernel(m,:);
    ini_ws=new_ws;
    new_ws = ini_ws - stepsize * grad(Y,ini_ws,kernel,lambda);
    
    %end time 
    time = toc(timeval);
    disp(time);
    
    if (time < 10)  
    cost=CostFunction(TrainingY,new_ws,training_kernel,lambda);
    result_p1(i,1)= time; 
    result_p1(i,2)= cost;
    i = i+1;
    disp(cost);
    end 
    
    itr=itr+1;
    disp(itr);
end

%delete all 0 costs 
result_p1( ~any(result_p1,2), : ) = [];

%figure 
figure;
plot(result_p1(:,1), result_p1(:,2));
title('Stochastic Gradient Descent with p=1');
xlabel('time');
ylabel('cost');

%calculate the accuracy 

  for j=1:n
        e= sigmoid(new_ws' * test_kernel(j,:)');
        if e >0.5 && TestY(j)==-1
            test_error_sg=test_error_sg+1;
        elseif e <= 0.5 && TestY(j)==1
            test_error_sg=test_error_sg+1;
        end       
  end
  accuracy=(1000-test_error_sg)/1000*100;

