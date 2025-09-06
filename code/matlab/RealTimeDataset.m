clc; clear; close all;
%% --- Step 1: Load new real-time dataset ---
newDataFile = 'abnormal_sensor_data.csv';  % current real-time data
newData = readtable(newDataFile);
%% --- Step 2: Load trained ML model ---
load('ML_SPOT.mat','model');  % previously trained Random Forest
%% --- Step 3: Take last 10 rows ---
numSamples = height(newData);
last10Rows = newData(max(1,numSamples-9):numSamples, :);  % last 10 samples
X_new = last10Rows{:,:};  % numeric values only
%% --- Step 4: Predict condition ---
[Y_pred, scores] = model.predict(X_new);
Y_pred = str2double(Y_pred);
%% --- Step 5: Display predictions ---
fprintf('Predictions for last 10 samples:\n');
for i = 1:length(Y_pred)
    state = 'NORMAL';
    if Y_pred(i) == 1
        state = 'ABNORMAL';
        beep;  % alert for abnormal
    end
    fprintf('Sample %d: %s\n', numSamples-10+i, state);
end
fprintf('Summary: %d ABNORMAL, %d NORMAL in last 10 samples\n', sum(Y_pred==1), sum(Y_pred==0));
%% --- Step 6: Plot waveforms for last 10 samples ---
numSensors = size(X_new,2);  % number of sensor columns
figure('Name','Last 10 Sensor Data Waveforms','NumberTitle','off');
for s = 1:numSensors
    subplot(numSensors,1,s);  % one plot per sensor
    plot(1:10, X_new(:,s), '-o','LineWidth',1.5);
    grid on;
    xlabel('Sample Number (Last 10)');
    ylabel(['Sensor ', num2str(s)]);
    title(['Waveform - Sensor ', num2str(s)]);
end
%% --- Step 7: Red alarm indicator if ABNORMAL count > NORMAL count ---
abnormalCount = sum(Y_pred == 1);
normalCount = sum(Y_pred == 0);

if abnormalCount > normalCount
    figure('Name', 'ABNORMALITY ALERT', 'Color', 'r', 'NumberTitle', 'off');
    text(0.5, 0.5, '!!! HIGH ABNORMALITY DETECTED !!!', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 18, 'FontWeight', 'bold', 'Color', 'w');
end
