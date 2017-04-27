%Chwan-Hao Tung
%861052182
%10/22/2016
%PS3 Q2

function [w,b] = learnsvm(X,Y,C)
m = size(X,1);
w=[0 0];
b=0;
summation = 0;
for i = 1:size(Y,1)
    summation = summation + (1-Y(i,1)*(w'*X(i,:))+b);
end
objective = C*summation+(1/2)*(w'*w);
stepsize = 1/(C*m);
while stepsize > (10^-6)/(C*m)
    for i = 1:100
        dw = 0;
        db = 0;
        for j = 1:size(Y,1)
            dw = dw + w;
            fx = (w*X'+b);
            classify = Y.*fx'; %get the classifiction of each data point.
            for k = 1:size(Y,1)
                if classify(k) < 1 %sum if 'incorrect'
                    dw=dw+C*(-Y(k,1)*X(k,:));
                    db=db+C*(-Y(k,1));
                end
            end
            if classify(j) < 1 %update w and b if point is 'bad'
                w = w - stepsize*dw;
                b = b - stepsize*db;
            end;
        end;
    end;
    summation = 0;
    for k = 1:size(Y,1)
        summation = summation + (1-Y(k,1)*(w'*X(k,:)+b));
    end
    testObj = C*summation+(1/2)*(w'*w);
    if testObj < objective %increase step size by 5% if objective gets better.
        stepsize = stepsize*1.05;
    else %decrease step size by 50% if objective gets worse.
        stepsize = stepsize*0.5;
    end
    objective = testObj;
end;

% dataset1 = X(Y==1,:);
% dataset2 = X(Y==-1,:);
% 
% plot(dataset1(:,1),dataset1(:,2),'bo');
% hold on;
% plot(dataset2(:,1),dataset2(:,2),'ro');
% drawline(w,b);
