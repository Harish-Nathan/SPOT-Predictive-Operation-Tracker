clc; clear; close all;

%% --- Step 1: Load datasets ---
normalData = readtable('normal_sensor_data.csv');
abnormalData = readtable('abnormal_sensor_data.csv');

%% --- Step 2: Add labels ---
normalData.Label = zeros(height(normalData),1);   % 0 = normal
abnormalData.Label = ones(height(abnormalData),1); % 1 = abnormal

%% --- Step 3: Combine datasets ---
% Make sure column names match exactly (case-sensitive)
abnormalData.Properties.VariableNames = normalData.Properties.VariableNames;
combinedData = [normalData; abnormalData];

%% --- Step 4: Shuffle and split ---
X = combinedData{:,1:end-1};
Y = combinedData.Label;

numSamples = size(X,1);
randIdx = randperm(numSamples);
X = X(randIdx,:);
Y = Y(randIdx);

trainRatio = 0.7;
numTrain = round(trainRatio * numSamples);

X_train = X(1:numTrain,:);
Y_train = Y(1:numTrain,:);
X_test  = X(numTrain+1:end,:);
Y_test  = Y(numTrain+1:end,:);

%% --- Step 5: Train model (Random Forest example) ---
numTrees = 100;
model = TreeBagger(numTrees, X_train, Y_train, 'Method', 'classification');

% Save trained model
save('ML_SPOT.mat','model');

%% --- Step 6: Test model ---
[Y_pred, scores] = model.predict(X_test);
Y_pred = str2double(Y_pred);

% Confusion Matrix
confMat = confusionmat(Y_test, Y_pred);
disp('Confusion Matrix:');
disp(confMat);

% Accuracy
accuracy = sum(Y_test == Y_pred) / numel(Y_test) * 100;
fprintf('Accuracy = %.2f%%\n', accuracy);