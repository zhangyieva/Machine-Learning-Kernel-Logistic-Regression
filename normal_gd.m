% Normal Gradient Descent
% 1. Test accuracy = 89.7000
% 2. Chosen step size = 0.01
% 3. Number of training iterations (until convergence or time limit)= 4166
% 4. Total training time (in seconds)= 289.3647

% load data 
load data1.mat

n=length(TrainingX);
%calculating ksquare 
Ksquare=0;
for i=1:n
    for j=1:n
        Ksquare=Ksquare+norm(TrainingX(i,:)-TrainingX(j,:))^2;
    end
end

Ksquare=Ksquare/n^2;

%calculating training kernel 
training_kernel=zeros(n,n);

for i=1:n
    for j=1:n
        training_kernel(i,j)=exp(-norm(TrainingX(i,:) - TrainingX(j,:))^2/Ksquare);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%
% Normal Gradient Descent%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%criteriens 
stepsize=0.01;
lambda=0.01;
epsilon=0.01;

%initial values 
itr=0;
test_error=0;
result_normal = zeros(50,2); 
ini_w=zeros(n,1);
i=1;

%calculating new w for 0s 
gradient = grad(TrainingY,ini_w,training_kernel,lambda);
new_w=ini_w - stepsize*gradient;

%start time 
tic
time_initial=tic; 

%staring loop
while norm(gradient) >= epsilon
    ini_w=new_w;
    gradient = grad(TrainingY,ini_w,training_kernel,lambda);
    new_w = ini_w - stepsize * gradient;
    timei = toc(time_initial);
    
   
    if (timei < 10)  
        cost=CostFunction(TrainingY,new_w,training_kernel,lambda);
        result_normal(i,2)= cost; 
        result_normal(i,1)= timei;
        i=i+1;
        disp(cost);
    end
    itr=itr+1;
    disp(itr);
end
total_time = toc(time_initial);

%delete all 0 costs 
result_normal( ~any(result_normal,2), : ) = [];

%graph 
figure;
plot(result_normal(:,1), result_normal(:,2));
title('Normal Gradient Descent');
xlabel('time');
ylabel('cost');

test_kernel=zeros(n,n);
for i=1:n
    for j=1:n
        test_kernel(i,j)=exp(-norm(TestX(i,:) - TrainingX(j,:))^2/Ksquare);
    end
end

    for j=1:n
        e= sigmoid(new_w' * test_kernel(j,:)');
        if e >0.5 && TestY(j)==-1
            test_error=test_error+1;
        elseif e <= 0.5 && TestY(j) ==1
            test_error=test_error+1;
        end       
    end
    
accuracy=(1000-test_error)/1000*100;

% call result
result_normal
accuracy
total_time
itr


