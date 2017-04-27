function q2
    f1 = figure('Name','example1.data','position',[0, 0, 1000, 1000]);
    f2 = figure('Name','example2.data','position',[0, 0, 1000, 1000]);

%     f2 = figure();
%     f3 = figure();
%     f4 = figure();
    data = load('example1.data','-ascii');
    Xdata = data(:,1:2);
    Ydata = data(:,3);
%     explot(Xdata,Ydata);
    figure(f1);
    subplot(3,3,1);
    [alpha,b] = learnsvm(Xdata,Ydata,0.1,@(A,B)A*B');
    w = calcW(alpha,Xdata,Ydata);
    plotclassifier(Xdata,Ydata,@(X) linearclassifier(X,w,b));
    title('C=0.1 with linear kernel');
    drawnow;
    
    subplot(3,3,2);
    [alpha,b] = learnsvm(Xdata,Ydata,1,@(A,B)A*B');
    w = calcW(alpha,Xdata,Ydata);
    plotclassifier(Xdata,Ydata,@(X) linearclassifier(X,w,b));
    title('C=1 with linear kernel');
    drawnow;
    
    subplot(3,3,3);
    [alpha,b] = learnsvm(Xdata,Ydata,10,@(A,B)A*B');
    w = calcW(alpha,Xdata,Ydata);
    plotclassifier(Xdata,Ydata,@(X) linearclassifier(X,w,b));
    title('C=10 with linear kernel');
    drawnow; 
    
    subplot(3,3,4);
    [alpha,b] = learnsvm(Xdata,Ydata,0.1,@(A,B) (A*B'+1).^2);
    plotclassifier(Xdata,Ydata,@(X) polyd2classifier(X,Xdata,Ydata,alpha,b));
    title('C=0.1 with poly kernel c=1 d=2');
    
    subplot(3,3,5);
    [alpha,b] = learnsvm(Xdata,Ydata,1,@(A,B) (A*B'+1).^2);
    plotclassifier(Xdata,Ydata,@(X) polyd2classifier(X,Xdata,Ydata,alpha,b));
    title('C=1 with poly kernel c=1 d=2');
    drawnow;
    
    subplot(3,3,6);
    [alpha,b] = learnsvm(Xdata,Ydata,10,@(A,B) (A*B'+1).^2);
    plotclassifier(Xdata,Ydata,@(X) polyd2classifier(X,Xdata,Ydata,alpha,b));
    title('C=10 with poly kernel c=1 d=2');
    drawnow;
       
    subplot(3,3,7);
    [alpha,b] = learnsvm(Xdata,Ydata,0.1,@(A,B) (A*B'+1).^5);
    plotclassifier(Xdata,Ydata,@(X) polyd5classifier(X,Xdata,Ydata,alpha,b));
    title('C=0.1 with poly kernel c=1 d=5');
    drawnow;
    
    subplot(3,3,8);
    [alpha,b] = learnsvm(Xdata,Ydata,1,@(A,B) (A*B'+1).^5);
    plotclassifier(Xdata,Ydata,@(X) polyd5classifier(X,Xdata,Ydata,alpha,b));
    title('C=1 with poly kernel c=1 d=5');
    drawnow;
    
    subplot(3,3,9);
    [alpha,b] = learnsvm(Xdata,Ydata,10,@(A,B) (A*B'+1).^5);
    plotclassifier(Xdata,Ydata,@(X) polyd5classifier(X,Xdata,Ydata,alpha,b));
    title('C=10 with poly kernel c=1 d=5');  
    drawnow;
    
    
    %Figure two
    data = load('example2.data','-ascii');
    Xdata = data(:,1:2);
    Ydata = data(:,3);
    figure(f2);
    subplot(3,3,1);
    [alpha,b] = learnsvm(Xdata,Ydata,0.1,@(A,B)A*B');
    w = calcW(alpha,Xdata,Ydata);
    plotclassifier(Xdata,Ydata,@(X) linearclassifier(X,w,b));
    title('C=0.1 with linear kernel');
    drawnow;
    
    subplot(3,3,2);
    [alpha,b] = learnsvm(Xdata,Ydata,1,@(A,B)A*B');
    w = calcW(alpha,Xdata,Ydata);
    plotclassifier(Xdata,Ydata,@(X) linearclassifier(X,w,b));
    title('C=1 with linear kernel');
    drawnow;
    
    subplot(3,3,3);
    [alpha,b] = learnsvm(Xdata,Ydata,10,@(A,B)A*B');
    w = calcW(alpha,Xdata,Ydata);
    plotclassifier(Xdata,Ydata,@(X) linearclassifier(X,w,b));
    title('C=10 with linear kernel');
    drawnow; 
    
    subplot(3,3,4);
    [alpha,b] = learnsvm(Xdata,Ydata,0.1,@(A,B) (A*B'+1).^2);
    plotclassifier(Xdata,Ydata,@(X) polyd2classifier(X,Xdata,Ydata,alpha,b));
    title('C=0.1 with poly kernel c=1 d=2');
    
    subplot(3,3,5);
    [alpha,b] = learnsvm(Xdata,Ydata,1,@(A,B) (A*B'+1).^2);
    plotclassifier(Xdata,Ydata,@(X) polyd2classifier(X,Xdata,Ydata,alpha,b));
    title('C=1 with poly kernel c=1 d=2');
    drawnow;
    
    subplot(3,3,6);
    [alpha,b] = learnsvm(Xdata,Ydata,10,@(A,B) (A*B'+1).^2);
    plotclassifier(Xdata,Ydata,@(X) polyd2classifier(X,Xdata,Ydata,alpha,b));
    title('C=10 with poly kernel c=1 d=2');
    drawnow;
       
    subplot(3,3,7);
    [alpha,b] = learnsvm(Xdata,Ydata,0.1,@(A,B) (A*B'+1).^5);
    plotclassifier(Xdata,Ydata,@(X) polyd5classifier(X,Xdata,Ydata,alpha,b));
    title('C=0.1 with poly kernel c=1 d=5');
    drawnow;
    
    subplot(3,3,8);
    [alpha,b] = learnsvm(Xdata,Ydata,1,@(A,B) (A*B'+1).^5);
    plotclassifier(Xdata,Ydata,@(X) polyd5classifier(X,Xdata,Ydata,alpha,b));
    title('C=1 with poly kernel c=1 d=5');
    drawnow;
    
    subplot(3,3,9);
    [alpha,b] = learnsvm(Xdata,Ydata,10,@(A,B) (A*B'+1).^5);
    plotclassifier(Xdata,Ydata,@(X) polyd5classifier(X,Xdata,Ydata,alpha,b));
    title('C=10 with poly kernel c=1 d=5');  
    drawnow;
    
    function w = calcW(alpha,Xdata,Ydata)
        w = zeros(2,1);
        for j =1:size(alpha)
            w(1) = w(1) + alpha(j)*Ydata(j)*Xdata(j,1);
            w(2) = w(2) + alpha(j)*Ydata(j)*Xdata(j,2);
        end
    end
    
    function Y = linearclassifier(X,w,b)
        Y = X*w+b;
    end
    function Y = polyd2classifier(X,Xdata, Ydata,alpha,b)
        Y = (((Xdata*X'+1).^2)'*(alpha.*Ydata)) + b;
    end
    function Y = polyd5classifier(X,Xdata, Ydata,alpha,b)
        Y = (((Xdata*X'+1).^5)'*(alpha.*Ydata)) + b;
    end
    
end