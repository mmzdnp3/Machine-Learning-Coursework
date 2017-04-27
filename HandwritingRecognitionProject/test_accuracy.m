%Chwan-Hao Tung
%861052182
%12/6
%CS229

function [accuracy] = test_accuracy(testFile)
    testData = load(testFile,'-ascii');
    Xtest = testData(:,2:end);
    Ytest = testData(:,1);
    Classifier = load('./Classifier.mat');
    dim = 32; % found by finding min of plot of testing error vs num dimensions.
    PCAtest = zeros(size(Xtest,1),dim);
    for i = 1:size(Xtest,1)
        PCAtest(i,:) = (Xtest(i,:)-Classifier.XTrainMean)*Classifier.coeff(:,1:dim);
    end
    predictionTest = predict(Classifier.Mdl, PCAtest);
    testError = sum(predictionTest~=Ytest)/length(Ytest);
    accuracy = (1 - testError)*100;
    fileID = fopen('./predictedLabels.txt','w');
    fprintf(fileID,'%u\n', predictionTest);
    fclose(fileID);
end


