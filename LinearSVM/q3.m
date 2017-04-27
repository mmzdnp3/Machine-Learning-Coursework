%Chwan-Hao Tung
%861052182
%10/22/2016
%PS3 Q3

%Takes about 2 mins to run!
function q3
f1 = figure;
xlabel('C');
ylabel('ErrorRates');

trainX = load('spamtrainX.data');
trainY = load('spamtrainY.data');
testX = load('spamtestX.data');
testY = load('spamtestY.data');
errorRates=[];
testingErrorRate=[];

Cs = logspace(-2,5,7); %7 C values from 10^-2 to 10^5

for C = Cs
    sumerrorR = 0;
    %3-fold cross-validation
    for i = 1:3
        testFoldX = trainX((i-1)*1000+1:i*1000,:);
        testFoldY = trainY((i-1)*1000+1:i*1000,:);
        if i == 1
            trainFoldX = trainX(i*1000+1:end,:);
            trainFoldY = trainY(i*1000+1:end,:);
        elseif i == 3
            trainFoldX = trainX(1:(i-1)*1000,:);
            trainFoldY = trainY(1:(i-1)*1000,:);
        else
            trainFoldX = vertcat(trainX(1:(i-1)*1000,:),trainX(i*1000+1:end,:));
            trainFoldY = vertcat(trainY(1:(i-1)*1000,:),trainY(i*1000+1:end,:));
        end
        [w, b] = qplearnsvm(trainFoldX,trainFoldY,C);
        classification = testFoldX*w + b;
        countIncorrect = 0;
        for j = 1:size(testFoldY)
            if(  testFoldY(j)*classification(j) < 0)  %count up the errors
                countIncorrect = countIncorrect +1;
            end    
        end
        sumerrorR = sumerrorR + countIncorrect/1000;
    end
    errorRates = [errorRates sumerrorR/3];
    sumerrorR = 0;
    classification = testX*w + b;
    countIncorrect = 0;
    for j = 1:size(testY)
        if(  testY(j)*classification(j) < 0)  %count up the errors
            countIncorrect = countIncorrect +1;
        end    
    end
    sumerrorR = sumerrorR + countIncorrect/1600;
    testingErrorRate = [testingErrorRate sumerrorR];
end
[M,I] = min(errorRates);
h1 = semilogx(Cs,errorRates);
hold on;
h2 = semilogx(Cs,testingErrorRate);
title(['Chosen C = ',num2str(Cs(I)),'. Testing accuracy of =',num2str(1-M)])
l = legend([h1 h2],{'CV Error','Test Error'});
set(l,'Fontsize',12);
