%Chwan-Hao Tung
%861052182
%10/13/2016
%(CS 229)
%(PS 2)%
tic;
f1 = figure;
f2 = figure;
xlabel('Lambda Values');
ylabel('Average Squared Error');
D = load('comm.txt','-ascii');
mseTrain = [];
mseTest = [];
aveErrorCV = [];
weights = [];
trainData = D(1:1000,:);
trainDataX = D(1:1000,1:end-1);
trainDataY = D(1:1000,end);
testDataX = D(1001:1994,1:end-1);
testDataY = D(1001:1994,end);
lambdas = logspace(-6,-1,15);
for lambda = lambdas
    mseCV1 = [];
    %Train Error
    [w,stats] = lasso(trainDataX,trainDataY,'Lambda',lambda);
    weights = [weights w];
    y = trainDataX*w + stats.Intercept;
    squarederrors = power(y-trainDataY,2);
    mseTrain = [mseTrain sum(squarederrors)/1000];
    %Test Error
    y = testDataX*w + stats.Intercept;
    squarederrors = power(y-testDataY,2);
    mseTest = [mseTest sum(squarederrors)/994];
    %10-fold cross validation
    for i = 1:10
        testFold = trainData((i-1)*100+1:i*100,1:end);
        if i == 1
            trainFolds = trainData(i*100+1:end,1:end);
        elseif i == 10
            trainFolds = trainData(1:(i-1)*100,1:end);
        else
            trainFolds = vertcat(trainData(1:(i-1)*100,1:end),trainData(i*100+1:end,1:end));
        end
        trainFoldX = trainFolds(:,1:end-1);
        trainFoldY = trainFolds(:,end);
        testFoldX = testFold(:,1:end-1);
        testFoldY = testFold(:,end);
        [w,stats] = lasso(trainFoldX,trainFoldY,'Lambda',lambda);
        y = testFoldX*w + stats.Intercept;
        squarederrors = power(y-testFoldY,2);
        mseCV1 = [mseCV1 sum(squarederrors)/100];
    end
    errorCV= sum(mseCV1)/10;
    aveErrorCV = [aveErrorCV errorCV];
end    
    
figure(f1);
h1 = semilogx(lambdas,mseTrain);
hold on;
h2 = semilogx(lambdas,mseTest);
hold on;
h3 = semilogx(lambdas,aveErrorCV);
l = legend([h1 h2 h3],{'Training Error','Testing Error','10FoldCV Error'});
set(l,'Fontsize',12);
figure(f2);
semilogx(lambdas,weights);
toc;
