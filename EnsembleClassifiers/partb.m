%Chwan-Hao Tung
%861052182
%CS229
%PS6
%partb
%11/15/2016
function partb
    data = load('class2d.ascii','-ascii');
    Xdata = data(:,1:2);
    Ydata = data(:,3);

    for i = 1:3
        fig = figure('position',[0, 0, 1000, 1000]);
        numDepth = i;
        alphas = ones(size(Xdata,1),1);
        weights = [];
        trees = cell(1000,1);

        t = traindtw(Xdata,Ydata,alphas,numDepth);
        trees(1) = {t};
        f = dt(Xdata, t);
        err = sum(alphas(Ydata ~= f))/sum(alphas);
        w = log((1-err)/err);
        weights = [weights w];
        alphas(Ydata ~= f) = alphas(Ydata ~= f)*exp(w);
        alphas = alphas/sum(alphas);

        subplot(2,2,1);
        plotclassifier(Xdata,Ydata, @(X) dt(X,t));
        drawnow;
        title([' Trees: 1, Depth:',num2str(numDepth)]);
        
        for j = 2:10
            t = traindtw(Xdata,Ydata,alphas,numDepth);
            trees(j) = {t};
            f = dt(Xdata,t);
            err = sum(alphas(Ydata ~= f))/sum(alphas);
            w = log((1-err)/err);
            weights = [weights w];
            alphas(Ydata ~= f) = alphas(Ydata ~= f)*exp(w);
            alphas = alphas/sum(alphas);
        end
        subplot(2,2,2);
        plotclassifier(Xdata,Ydata, @(X) myclassifier(X,trees,weights,10));
        drawnow;
        title([' Trees: 10, Depth:',num2str(numDepth)]);

        for j = 11:100
            t = traindtw(Xdata,Ydata,alphas,numDepth);
            trees(j) = {t};
            f = dt(Xdata,t);
            err = sum(alphas(Ydata ~= f))/sum(alphas);
            w = log((1-err)/err);
            weights = [weights w];
            alphas(Ydata ~= f) = alphas(Ydata ~= f)*exp(w);
            alphas = alphas/sum(alphas);
        end
        subplot(2,2,3);
        plotclassifier(Xdata,Ydata, @(X) myclassifier(X,trees,weights,100));
        drawnow;
        title([' Trees: 100, Depth:',num2str(numDepth)]);
        
        for j = 101:1000
            t = traindtw(Xdata,Ydata,alphas,numDepth);
            trees(j) = {t};
            f = dt(Xdata,t);
            err = sum(alphas(Ydata ~= f))/sum(alphas);
            w = log((1-err)/err);
            weights = [weights w];
            alphas(Ydata ~= f) = alphas(Ydata ~= f)*exp(w);
            alphas = alphas/sum(alphas);
        end
        subplot(2,2,4);
        plotclassifier(Xdata,Ydata, @(X) myclassifier(X,trees,weights,1000));
        drawnow;
        title([' Trees: 1000, Depth:',num2str(numDepth)]);
    end   
end
function Y = myclassifier(X,trees,weights,numTrees)
    Y = zeros(size(X,1),1);
    for k = 1:numTrees
        Y = Y + weights(k)*dt(X,trees{k});
    end
    Y = sign(Y);
end

