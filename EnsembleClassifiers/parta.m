%Chwan-Hao Tung
%861052182
%CS229
%PS6
%parta
%11/15/2016
function parta
    data = load('class2d.ascii','-ascii');
    Xdata = data(:,1:2);
    Ydata = data(:,3);

    for i = 1:3
        fig = figure('position',[0, 0, 1000, 1000]);
        %for 1 tree
        numDepth = i;
        sample = data(randi(size(data,1),size(data,1),1),:);
        Xsample = sample(:,1:2);
        Ysample = sample(:,3);
        t = traindt(Xsample,Ysample,numDepth);
        subplot(2,2,1);
        plotclassifier(Xdata,Ydata, @(X) dt(X,t));
        drawnow;
        title([' Trees: 1, Depth:',num2str(numDepth)]);
        %for 10 trees
        trees = cell(1000,1);
        trees(1) = {t};

        for j = 2:10
            sample = data(randi(size(data,1),size(data,1),1),:);
            Xsample = sample(:,1:2);
            Ysample = sample(:,3);
            trees(j) = {traindt(Xsample,Ysample,numDepth)};
        end
        subplot(2,2,2);
        plotclassifier(Xdata,Ydata, @(X) myclassifier(X,trees,10));
        drawnow;
        title([' Trees: 10, Depth:',num2str(numDepth)]);
        %for 100 trees
        for j = 11:100
            sample = data(randi(size(data,1),size(data,1),1),:);
            Xsample = sample(:,1:2);
            Ysample = sample(:,3);
            trees(j) = {traindt(Xsample,Ysample,numDepth)};
        end
        subplot(2,2,3);
        plotclassifier(Xdata,Ydata, @(X) myclassifier(X,trees,100));    
        title([' Trees: 100, Depth:',num2str(numDepth)]);
        %for 1000 trees
        for j = 101:1000
            sample = data(randi(size(data,1),size(data,1),1),:);
            Xsample = sample(:,1:2);
            Ysample = sample(:,3);
            trees(j) = {traindt(Xsample,Ysample,numDepth)};
        end
        subplot(2,2,4);
        plotclassifier(Xdata,Ydata, @(X) myclassifier(X,trees,1000));
        drawnow;
        title([' Trees: 1000, Depth:',num2str(numDepth)]);
    end
end
function Y = myclassifier(X,trees,numTrees)
    Y = zeros(size(X,1),1);
    for k = 1:numTrees
        Y = Y + dt(X,trees{k});
    end
    Y = sign(Y);
end