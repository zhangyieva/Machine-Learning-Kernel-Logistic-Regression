
%all 3 graphs 
figure 
xPlot=plot(result_normal(:,1), result_normal(:,2),'b'); 
hold on 
yPlot=plot(result_p100(:,1), result_p100(:,2),'r');
zPlot=plot(result_p1(:,1), result_p1(:,2),'g');
xlabel('Time');
ylabel('Cost');
title('Cost Functions over Time of the Three Methods');

xName = 'Normal Gradient Descent';
yName = 'SGD P=100';
zName = 'SGD P=1';
legend(xName,yName,zName);
hold off