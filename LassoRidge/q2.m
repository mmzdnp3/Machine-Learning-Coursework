%Chwan-Hao Tung
%861052182
%10/13/2016
%(CS 229)
%(PS 2)%



f1 = figure;
f2 = figure;
f3 = figure;
accurateX = -pi:0.01:pi;
y =tan(pi*accurateX/3)+(accurateX-0.5).^2;

lambda1 = 0.001;
lambda2 = 0.1;
lambda3 = 10;


figure(f1);
yhats = [];
for i = 1:100
    X = -1 + (1+1)*rand(10,1);
    X = sort(X);
    Y =tan(pi*X/3)+(X-0.5).^2;
    phiX = [ones(10,1) X X.*X X.*X.*X X.*X.*X.*X X.*X.*X.*X.*X];
    W = inv(phiX'*phiX+eye(size(phiX'*phiX))*lambda1)*phiX'*(Y+0.5*randn(10,1));
    X = linspace(-1,1);
    X = X';
    phiX = [ones(100,1) X X.*X X.*X.*X X.*X.*X.*X X.*X.*X.*X.*X];
    yhat = phiX*W;
    yhats = [yhats yhat];
    plot(X,yhat,'r','Linewidth',0.1);
    hold on;
end
aveY = mean(yhats,2);
plot(X, aveY,'b','Linewidth',2);
plot(accurateX,y,'k','Linewidth',2);
title('lambda = 0.001');
axis([-1 1 -0.5 4.5]);

figure(f2);
yhats = [];
for i = 1:100
    X = -1 + (1+1)*rand(10,1);
    X = sort(X);
    Y =tan(pi*X/3)+(X-0.5).^2;
    phiX = [ones(10,1) X X.*X X.*X.*X X.*X.*X.*X X.*X.*X.*X.*X];
    W = inv(phiX'*phiX+eye(size(phiX'*phiX))*lambda2)*phiX'*(Y+0.5*randn(10,1));
    X = linspace(-1,1);
    X = X';
    phiX = [ones(100,1) X X.*X X.*X.*X X.*X.*X.*X X.*X.*X.*X.*X];
    yhat = phiX*W;
    yhats = [yhats yhat];
    plot(X,yhat,'r','Linewidth',0.1);
    hold on;
end
aveY = mean(yhats,2);
plot(X, aveY,'b','Linewidth',2);
plot(accurateX,y,'k','Linewidth',2);
title('lambda = 0.1');
axis([-1 1 -0.5 4.5]);


figure(f3);
yhats = [];
for i = 1:100
    X = -1 + (1+1)*rand(10,1);
    X = sort(X);
    Y =tan(pi*X/3)+(X-0.5).^2;
    phiX = [ones(10,1) X X.*X X.*X.*X X.*X.*X.*X X.*X.*X.*X.*X];
    W = inv(phiX'*phiX+eye(size(phiX'*phiX))*lambda3)*phiX'*(Y+0.5*randn(10,1));
    X = linspace(-1,1);
    X = X';
    phiX = [ones(100,1) X X.*X X.*X.*X X.*X.*X.*X X.*X.*X.*X.*X];
    yhat = phiX*W;
    yhats = [yhats yhat];
    plot(X,yhat,'r','Linewidth',0.1);
    hold on;
end
aveY = mean(yhats,2);
plot(X, aveY,'b','Linewidth',2);
plot(accurateX,y,'k','Linewidth',2);
title('lambda = 10');
axis([-1 1 -0.5 4.5]);