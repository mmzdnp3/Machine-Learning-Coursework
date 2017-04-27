%Chwan-Hao Tung
%861052182
%11/27 2016
%CS229
%PS7

tic;
traindata = load('spamtrain.ascii','-ascii');
testdata = load('spamtest.ascii','-ascii');

Xtrain = traindata(:,1:end-1); 
Ytrain = traindata(:,end);
Xtest = testdata(:,1:end-1);
Ytest = testdata(:,end);

rounds = floor(logspace(0,2+log10(5),10));

%for bagging

fig = figure('position',[0, 0, 1000, 1000]);
subplot(2,2,1);
trees = cell(1000,1);
index = 1;
trainerrors = [];
testerrors = [];
for round = rounds
    for k = index:round
        numDepth = 2;
        sample = traindata(randi(size(traindata,1),size(traindata,1),1),:);
        Xsample = sample(:,1:end-1);
        Ysample = sample(:,end);
        t = traindt(Xsample,Ysample,numDepth);
        trees(index) = {t};
        index = index +1;
    end
    Y = zeros(size(Xtest,1),1);
    for k = 1:round
        Y = Y + dt(Xtest,trees{k});
    end
    Y = sign(Y);
    error = sum(Ytest ~= Y)/size(Ytest,1);
    testerrors = [testerrors error];
end
semilogx(rounds, testerrors, 'Linewidth', 2);
hold on;


lambda = logspace(-4,-1,10);
Wtrain = zeros(size(Xtrain,1),round);
Wtest = zeros(size(Xtest,1),round);
testerrors = zeros(size(lambda));
for i = 1:round
    Wtrain(:,i) = dt(Xtrain,trees{i});
end
[w, other ] = lassoglm (Wtrain,Ytrain==1, 'binomial' , 'Standardize' ,0 , 'Lambda' , lambda );
for l = 1:size(lambda,2)
    % lambda is a lambda value , or a vector of lambda values
    w_ = 2*w(:,l); % weight vector , or weight matrix ( one column for each lambda value )
    w0 = 2*other.Intercept(l) - 1; % bias term or vector of same ( one for each lambda)
    for i = 1:round
        Wtest(:,i) = dt(Xtest,trees{i});
    end
    Y = Wtest*w_ + w0;
    Y = sign(Y);
    error = sum(Ytest ~= Y)/size(Ytest,1);
    testerrors(l) = error;
end
numtrees = sum(w ~= 0);
semilogx(numtrees,testerrors, 'Linewidth', 2);
hold on;
title('Testing Error - Bagging');
xlabel('Rounds');
ylabel('ErrorRates');   
legend('Bagging','Bagging Reweighted');

%for boosting
subplot(2,2,2);
index = 1;
alphas = ones(size(Xtrain,1),1);
weights = [];
trees = cell(1000,1);
trainerrors = [];
testerrors = [];
for round = rounds
    for k = index:round
        numDepth = 2;
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
    Y = zeros(size(Xtest,1),1);
    for k = 1:round
        Y = Y + weights(k)*dt(Xtest,trees{k});
    end
    Y = sign(Y);
    error = sum(Ytest ~= Y)/size(Ytest,1);
    testerrors = [testerrors error];
end
semilogx(rounds, testerrors, 'Linewidth', 2);
hold on;

lambda = logspace(-4,-1,10);
Wtrain = zeros(size(Xtrain,1),round);
Wtest = zeros(size(Xtest,1),round);
testerrors = zeros(size(lambda));
for i = 1:round
    Wtrain(:,i) = dt(Xtrain,trees{i});
end
[w, other ] = lassoglm (Wtrain,Ytrain==1, 'binomial' , 'Standardize' ,0 , 'Lambda' , lambda );
for l = 1:size(lambda,2)
    % lambda is a lambda value , or a vector of lambda values
    w_ = 2*w(:,l); % weight vector , or weight matrix ( one column for each lambda value )
    w0 = 2*other.Intercept(l) - 1; % bias term or vector of same ( one for each lambda)
    for i = 1:round
        Wtest(:,i) = dt(Xtest,trees{i});
    end
    Y = Wtest*w_ + w0;
    Y = sign(Y);
    error = sum(Ytest ~= Y)/size(Ytest,1);
    testerrors(l) = error;
end
numtrees = sum(w ~= 0);
semilogx(numtrees,testerrors, 'Linewidth', 2);
hold on;
title('Testing Error - Boosting ');
xlabel('Rounds');
ylabel('ErrorRates');   
legend('Boosting','Boosting Reweighted');

toc;


