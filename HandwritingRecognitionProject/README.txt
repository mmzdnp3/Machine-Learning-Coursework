Chwan-Hao Tung
861052182
CS229

Files:
	train_classifier.m	- matlab function to train classifier on handwriting.data		
	test_accuracy.m		- matlab function to test accuracy	
	Classifier.mat		- classifier created by train_classifier.mat, used in test_accuracy.m.
	handwriting.data	- data given to us to train classifier on.
	
Function descriptions:
	//Retrains the classifier using training data from handwriting.data
	train_classifier 
	
		Creates a .mat file named 'Classifier.mat' in the current MATLAB path.
		Used for test_accuracy.

	//Tests classifier accuracy against testing examples. 
	test_accuracy(testFilename)
		-testFilename: file path containing testing examples.
		
		This function requires Classifier.mat to be in the current MATLAB path.
		Returns accuracy in % according to the labels in testFilename. 
		Creates text file named 'predictLabels.txt' in the current MATLAB path.
		'predictLabels.txt' contains the labels of the predictions from the classifier. 1 label per line.
		
		The labels can be loaded with
			load predictedLabels.txt -ascii

Note:
		To run method on new data without retraining. The packaged Classifier.mat must be moved to the current MATLAB path. 
		
Example:
	>> train_classifier
	>> test_accuracy('testData')

	ans =

	   87.0791

	>> 

Testing environment:
	Matlab version R2016b (9.1.0.441655)
	
	