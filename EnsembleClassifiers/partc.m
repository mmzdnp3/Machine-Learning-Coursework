%Chwan-Hao Tung
%861052182
%CS229
%PS6
%partc
%11/15/2016
tic;
traindata = load('spamtrain.ascii','-ascii');
testdata = load('spamtest.ascii','-ascii');

Xtrain = traindata(:,1:end-1); 
Ytrain = traindata(:,end);
Xtest = testdata(:,1:end-1);
Ytest = testdata(:,end);

rounds = floor(logspace(0,3,10));

%for bagging

fig = figure('position',[0, 0, 1000, 1000]);
for i = 1:3
    trees = cell(1000,1);
    index = 1;
    trainerrors = [];
    testerrors = [];
    for round = rounds
        for k = index:round
            numDepth = i;
            sample = traindata(randi(size(traindata,1),size(traindata,1),1),:);
            Xsample = sample(:,1:end-1);
            Ysample = sample(:,end);
            t = traindt(Xsample,Ysample,numDepth);
            trees(index) = {t};
            index = index +1;
        end
        Y = zeros(size(Xtrain,1),1);
        for k = 1:round
            Y = Y + dt(Xtrain,trees{k});
        end
        Y = sign(Y);
        error = sum(Ytrain ~= Y)/size(Ytrain,1);
        trainerrors = [trainerrors error];
        Y = zeros(size(Xtest,1),1);
        for k = 1:round
            Y = Y + dt(Xtest,trees{k});
        end
        Y = sign(Y);
        error = sum(Ytest ~= Y)/size(Ytest,1);
        testerrors = [testerrors error];
    end
    semilogx(rounds, trainerrors,'--','Linewidth', i);
    hold on;
    semilogx(rounds, testerrors, 'Linewidth', i);
    hold on;
    title('Bagging');
    xlabel('Rounds');
    ylabel('ErrorRates');   
end
legend('Train Depth 1','Test Depth 1','Train Depth 2','Test Depth 2','Train Depth 3','Test Depth 3');

% %for boosting
fig = figure('position',[0, 0, 1000, 1000]);
for i = 1:3
    index = 1;
    alphas = ones(size(Xtrain,1),1);
    weights = [];
    trees = cell(1000,1);
    trainerrors = [];
    testerrors = [];
    for round = rounds
        for k = index:round
            numDepth = i;
            t = traindtw(Xtrain,Ytrain,alphas,numDepth);
            f = dt(Xtrain,t);
            err = sum(alphas(Ytrain ~= f))/sum(alphas);
            w = log((1-err)/err);
            weights = [weights w];
            alphas(Ytrain ~= f) = alphas(Ytrain ~= f)*exp(w);
            alphas = alphas/sum(alphas);
            trees(index) = {t};
            index = index +1;
        end
        Y = zeros(size(Xtrain,1),1);
        for k = 1:round
            Y = Y + weights(k)*dt(Xtrain,trees{k});
        end
        Y = sign(Y);
        error = sum(Ytrain ~= Y)/size(Ytrain,1);
        trainerrors = [trainerrors error];
        Y = zeros(size(Xtest,1),1);
        for k = 1:round
            Y = Y + weights(k)*dt(Xtest,trees{k});
        end
        Y = sign(Y);
        error = sum(Ytest ~= Y)/size(Ytest,1);
        testerrors = [testerrors error];
    end
    semilogx(rounds, trainerrors,'--','Linewidth', i);
    hold on;
    semilogx(rounds, testerrors, 'Linewidth', i);
    hold on;
    title('Boosting');
    xlabel('Rounds');
    ylabel('ErrorRates');  
end
legend('Train Depth 1','Test Depth 1','Train Depth 2','Test Depth 2','Train Depth 3','Test Depth 3');
toc;


