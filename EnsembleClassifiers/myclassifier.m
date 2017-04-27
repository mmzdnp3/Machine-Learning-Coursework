function Y = myclassifier(X,Ydata,D,numTrees,numDepth)
    Y = zeros(size(X));
    size(Y)
    for j = 1:numTrees
        sample = D(randperm(size(D,1)),:);
        t = traindt(sample,Ydata,numDepth);
        size(dt(X,t))
        Y = Y + dt(X,t);
    end
end