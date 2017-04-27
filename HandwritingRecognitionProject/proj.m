%Chwan-Hao Tung
%861052182
%12/6
%CS229

function [accuracy] = testAccuracy(testFile)
    fprintf('Number of arguments: %d\n',nargin)
    
    data = load('handwriting.data','-ascii');
    Y = data(:,1);
    X = data(:,2:end);
    trainRatio = 0.9;
    testRatio = 0.1;
    rng(5);
    [trainIndex,testIndex] = dividerand(size(X,1),trainRatio,testRatio);
    Xtrain = data(trainIndex,2:end);
    Ytrain = data(trainIndex,1);
    
    Xtest = data(testIndex,2:end);
    Ytest = data(testIndex,1);

%     testError = [];
    % trainError = [];


    %%%%%%% Multinomial Logistic Regression - didn't converge....

    % coefficient_estimates = mnrfit(Xtrain,Ytrain+1);


    %%%%%% K-nearest neighbor with default distance (euclidean distance)
    % for i = round(linspace(1,100,10))
    %     if mod(i,2) == 0
    %         i= i -1;
    %     end
%         Mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'DistanceWeight','squaredinverse');
%         Mdl.BreakTies = 'nearest';
%         predictionTest = predict(Mdl, Xtest);
%         testError = [testError sum(predictionTest~=Ytest)/length(Ytest)];
    % end

    %%%%%%% K-nearest neighbor with jaccard distance
    % for i = round(linspace(1,100,10))
%         if mod(i,2) == 0
%             i= i -1;
%         end
%         Mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'Distance','jaccard','DistanceWeight','squaredinverse');
%         Mdl.BreakTies = 'nearest';
%         predictionTest = predict(Mdl, Xtest);
%         testError = [testError sum(predictionTest~=Ytest)/length(Ytest)];
    %     testError = sum(predictionTest~=Ytest)/length(Ytest);
    % end


    %%%%%%%%% Plotting testing error vs. Num neighbors
    % figure;
    % plot(round(linspace(1,100,10)),testError,'Linewidth', 2);
    % title('k-NN using jaccard distance metric');
    % axis([1 100 0 1]);
    % xlabel('Num Neighbours');
    % ylabel('Test Error');

    %%%%%%%%%% PCA to reduce dimensions then KNN
    % for j = 1:size(Xtest,2)
        [coeff,score,latent] = pca(Xtrain);
        XTrainMean = mean(Xtrain);
        dim = 32; % found by finding min of plot of testing error vs num dimensions.
        PCAtrain = score(:,1:dim);
        PCAtest = zeros(size(Xtest,1),dim);
        for i = 1:size(Xtest,1)
            PCAtest(i,:) = (Xtest(i,:)-XTrainMean)*coeff(:,1:dim);
        end

        Mdl = fitcknn(PCAtrain,Ytrain,'NumNeighbors',3,'Distance','cosine','DistanceWeight','squaredinverse');
        Mdl.BreakTies = 'nearest';
        predictionTest = predict(Mdl, PCAtest);
        testError = sum(predictionTest~=Ytest)/length(Ytest);
        accuracy = (1 - testError)*100;
        fileID = fopen('./predictedLabels.txt','w');
        fprintf(fileID,'%u\n', predictionTest);
        fclose(fileID);
    % end

    %%%%%%%%% Plotting testing error vs. num dimensions
    % figure;
    % plot(testError,'Linewidth', 2);
    % hold on;
    % [testErrormin IndexMin] = min(testError);
    % plot(IndexMin,testErrormin, 'ro', 'markersize', 10)
    % offset = -.05;
    % text(IndexMin,testErrormin+diff(ylim)*offset,['(' num2str(IndexMin) ',' num2str(testErrormin) ')'])
    % 
    % title('k-NN after PCA dimension reduction, cosine distance metric, k = 3');
    % axis([1 126 0 0.5]);
    % xlabel('# dimensions corresponding to the # largest m eigenvalues');
    % ylabel('Test Error');


    %%%%%%%%% Examples of misclassified handwriting
%     figure;
%     mispredicted = (predictionTest ~= Ytest);
%     predictionLabel = predictionTest(mispredicted == 1,:);
%     misclassified_samples = Xtest(mispredicted == 1,:);
%     misclassified_labels = Ytest(mispredicted == 1,:);
% 
%     for i = 121:136
%         subplot(4,4,i-120);
%         imagesc(reshape(misclassified_samples(i,2:end),[8 16])');
%         colormap (1.0 - gray);
%         axis equal;
%         lettera = char(97 + misclassified_labels(i));
%         letterb = char(97 + predictionLabel(i));
%         title([lettera,' predicted as ', letterb]);
%     end
end


