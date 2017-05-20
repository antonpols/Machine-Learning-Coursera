clear;
close all;
p = gcp('nocreate');
if isempty(p)
    parpool();
end

%% load the preprocced data

load('preprocessed_total_dataset.mat');
load('trained_models.mat');

%% crossvalidation SVM


%linear kernel SVM

% use given SVM predict function
predictions_validation_given_linear = cell(numel(C),1);
accuracy_validation_given_linear = NaN*zeros(numel(C),1);
f_score_validation_given_linear = NaN*zeros(numel(C),1);
parfor_progress(numel(C));
parfor i = 1:numel(C)
    predictions_validation_given_linear{i} = svmPredict(model_array_given_linear{i},validation_set_feature_vectors);
    accuracy_validation_given_linear(i) = mean(predictions_validation_given_linear{i} == validation_set_labels);
    true_positive = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_given_linear{i} == 1)));
    false_positive = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_given_linear{i} == 1)));
    false_negative = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_given_linear{i} == 0)));
    true_negative = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_given_linear{i} == 0)));
    precision = true_positive/(true_positive+false_positive);
    recall = true_positive/(true_positive+false_negative);
    f_score_validation_given_linear(i) = 2*(precision*recall)/(precision+recall);
    parfor_progress;
end
parfor_progress(0);

% use libsvm predict function
predictions_validation_libSVM_linear = cell(numel(C),1);
accuracy_validation_libSVM_linear = NaN*zeros(numel(C),1);
f_score_validation_libSVM_linear = NaN*zeros(numel(C),1);
parfor_progress(numel(C));
parfor i = 1:numel(C)
    predictions_validation_libSVM_linear{i} = svmpredict(validation_set_labels,validation_set_feature_vectors,model_array_libSVM_linear{i},'-q');
    accuracy_validation_libSVM_linear(i) = mean(predictions_validation_libSVM_linear{i} == validation_set_labels);
    true_positive = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_libSVM_linear{i} == 1)));
    false_positive = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_libSVM_linear{i} == 1)));
    false_negative = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_libSVM_linear{i} == 0)));
    true_negative = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_libSVM_linear{i} == 0)));
    precision = true_positive/(true_positive+false_positive);
    recall = true_positive/(true_positive+false_negative);
    f_score_validation_libSVM_linear(i) = 2*(precision*recall)/(precision+recall);
    parfor_progress;
end
parfor_progress(0);


figure;
subplot(2,1,1);
semilogx(C,accuracy_validation_given_linear,'xb');
hold on;
semilogx(C,accuracy_validation_libSVM_linear,'xr');
legend('given functions','libSVM')
xlabel('C');
ylabel('Accuracy');
title('Accuracy of crossvalidations models obtained by SVM with linear kernel')

subplot(2,1,2);
semilogx(C,f_score_validation_given_linear,'xb');
hold on;
semilogx(C,f_score_validation_libSVM_linear,'xr');
legend('given functions','libSVM')
xlabel('C');
ylabel('F1 score');
title('F1 score of crossvalidations models obtained by SVM with linear kernel')


%gaussian kernel SVM

% use given SVM predict function
predictions_validation_given_gaussian = cell(1,numel(C));
accuracy_validation_given_gaussian = NaN*zeros(numel(C),numel(sigma));
f_score_validation_given_gaussian = NaN*zeros(numel(C),numel(sigma));
parfor_progress(numel(C));
parfor i = 1:numel(C)
    predictions_validation_given_gaussian_temp = cell(1,numel(sigma));
    accuracy_validation_given_gaussian_temp = NaN*zeros(1,numel(sigma));
    f_score_validation_given_gaussian_temp = NaN*zeros(1,numel(sigma));
    for j = 1:numel(sigma)
        predictions_validation_given_gaussian_temp{j} = svmPredict(model_array_given_gaussian{i}{j},validation_set_feature_vectors);
        accuracy_validation_given_gaussian_temp(j) = mean(predictions_validation_given_gaussian_temp{j} == validation_set_labels);
        true_positive = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_given_gaussian_temp{j} == 1)));
        false_positive = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_given_gaussian_temp{j} == 1)));
        false_negative = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_given_gaussian_temp{j} == 0)));
        true_negative = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_given_gaussian_temp{j} == 0)));
        precision = true_positive/(true_positive+false_positive);
        recall = true_positive/(true_positive+false_negative);
        f_score_validation_given_gaussian_temp(j) = 2*(precision*recall)/(precision+recall);
    end
    predictions_validation_given_gaussian{i} = predictions_validation_given_gaussian_temp;
    accuracy_validation_given_gaussian(i,:) = accuracy_validation_given_gaussian_temp;
    f_score_validation_given_gaussian(i,:) = f_score_validation_given_gaussian_temp;
    parfor_progress;
end
parfor_progress(0);

% use libsvm predict function
predictions_validation_libSVM_gaussian = cell(1,numel(C));
accuracy_validation_libSVM_gaussian = NaN*zeros(numel(C),numel(sigma));
f_score_validation_libSVM_gaussian = NaN*zeros(numel(C),numel(sigma));
parfor_progress(numel(C));
parfor i = 1:numel(C)
    predictions_validation_libSVM_gaussian_temp = cell(1,numel(sigma));
    accuracy_validation_libSVM_gaussian_temp = NaN*zeros(1,numel(sigma));
    f_score_validation_libSVM_gaussian_temp = NaN*zeros(1,numel(sigma));
    for j = 1:numel(sigma)
        predictions_validation_libSVM_gaussian_temp{j} = svmpredict(validation_set_labels,validation_set_feature_vectors,model_array_libSVM_gaussian{i}{j},'-q');
        accuracy_validation_libSVM_gaussian_temp(j) = mean(predictions_validation_libSVM_gaussian_temp{j} == validation_set_labels);
        true_positive = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_libSVM_gaussian_temp{j} == 1)));
        false_positive = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_libSVM_gaussian_temp{j} == 1)));
        false_negative = sum(ismember(find(validation_set_labels == 1),find(predictions_validation_libSVM_gaussian_temp{j} == 0)));
        true_negative = sum(ismember(find(validation_set_labels == 0),find(predictions_validation_libSVM_gaussian_temp{j} == 0)));
        precision = true_positive/(true_positive+false_positive);
        recall = true_positive/(true_positive+false_negative);
        f_score_validation_libSVM_gaussian_temp(j) = 2*(precision*recall)/(precision+recall);
    end
    predictions_validation_libSVM_gaussian{i} = predictions_validation_libSVM_gaussian_temp;
    accuracy_validation_libSVM_gaussian(i,:) = accuracy_validation_libSVM_gaussian_temp;
    f_score_validation_libSVM_gaussian(i,:) = f_score_validation_libSVM_gaussian_temp;
    parfor_progress;
end
parfor_progress(0);


[C_grid, sigma_grid] = meshgrid(C,sigma);

figure;
surf(C_grid,sigma_grid,accuracy_validation_given_gaussian');
set(gca, 'XScale', 'log', 'YScale', 'log', 'ZScale', 'log')
xlabel('C');
ylabel('Sigma');
zlabel('Accuracy');
title('Accuracy of crossvalidations models obtained by given SVM with Gaussian kernel')

figure;
surf(C_grid,sigma_grid,accuracy_validation_libSVM_gaussian');
set(gca, 'XScale', 'log', 'YScale', 'log', 'ZScale', 'log')
xlabel('C');
ylabel('Sigma');
zlabel('Accuracy');
title('Accuracy of crossvalidations models obtained by libSVM with Gaussian kernel')

figure;
surf(C_grid,sigma_grid,f_score_validation_given_gaussian');
set(gca, 'XScale', 'log', 'YScale', 'log', 'ZScale', 'log')
xlabel('C');
ylabel('Sigma');
zlabel('F1 score');
title('F1 score of crossvalidations models obtained by given SVM with Gaussian kernel')

figure;
surf(C_grid,sigma_grid,f_score_validation_libSVM_gaussian');
set(gca, 'XScale', 'log', 'YScale', 'log', 'ZScale', 'log')
xlabel('C');
ylabel('Sigma');
zlabel('F1 score');
title('F1 score of crossvalidations models obtained by libSVM with Gaussian kernel')


%% test SVM


%linear kernel SVM

% use given SVM predict function
[~,index_max_given_linear] = max(f_score_validation_given_linear);
predictions_test_given_linear = svmPredict(model_array_given_linear{index_max_given_linear},test_set_feature_vectors);
accuracy_test_given_linear = mean(predictions_test_given_linear == test_set_labels);
true_positive = sum(ismember(find(test_set_labels == 1),find(predictions_test_given_linear == 1)));
false_positive = sum(ismember(find(test_set_labels == 0),find(predictions_test_given_linear == 1)));
false_negative = sum(ismember(find(test_set_labels == 1),find(predictions_test_given_linear == 0)));
true_negative = sum(ismember(find(test_set_labels == 0),find(predictions_test_given_linear == 0)));
precision = true_positive/(true_positive+false_positive);
recall = true_positive/(true_positive+false_negative);
f_score_test_given_linear = 2*(precision*recall)/(precision+recall);

fprintf('Using given SVM train and predict functions and the linear kernel, we obtain: \nAccuracy: %0.4f \nF1 score: %0.4f \n',accuracy_test_given_linear,f_score_test_given_linear);

% use libsvm predict function
[~,index_max_libSVM_linear] = max(f_score_validation_libSVM_linear);
predictions_test_libSVM_linear = svmpredict(test_set_labels,test_set_feature_vectors,model_array_libSVM_linear{index_max_libSVM_linear},'-q');
accuracy_test_libSVM_linear = mean(predictions_test_libSVM_linear == test_set_labels);
true_positive = sum(ismember(find(test_set_labels == 1),find(predictions_test_libSVM_linear == 1)));
false_positive = sum(ismember(find(test_set_labels == 0),find(predictions_test_libSVM_linear == 1)));
false_negative = sum(ismember(find(test_set_labels == 1),find(predictions_test_libSVM_linear == 0)));
true_negative = sum(ismember(find(test_set_labels == 0),find(predictions_test_libSVM_linear == 0)));
precision = true_positive/(true_positive+false_positive);
recall = true_positive/(true_positive+false_negative);
f_score_test_libSVM_linear = 2*(precision*recall)/(precision+recall);

fprintf('Using libSVM and the linear kernel, we obtain: \nAccuracy: %0.4f \nF1 score: %0.4f \n',accuracy_test_libSVM_linear,f_score_test_libSVM_linear);


%gaussian kernel SVM

% use given SVM predict function
[f_score_validation_given_gaussian_max_row, index_max_given_gaussian_max_row] = max(f_score_validation_given_gaussian);
[~,index_max_given_gaussian_max_column] = max(f_score_validation_given_gaussian_max_row);
predictions_test_given_gaussian = svmPredict(model_array_given_gaussian{index_max_given_gaussian_max_row(index_max_given_gaussian_max_column)}{index_max_given_gaussian_max_column},test_set_feature_vectors);
accuracy_test_given_gaussian = mean(predictions_test_given_gaussian == test_set_labels);
true_positive = sum(ismember(find(test_set_labels == 1),find(predictions_test_given_gaussian == 1)));
false_positive = sum(ismember(find(test_set_labels == 0),find(predictions_test_given_gaussian == 1)));
false_negative = sum(ismember(find(test_set_labels == 1),find(predictions_test_given_gaussian == 0)));
true_negative = sum(ismember(find(test_set_labels == 0),find(predictions_test_given_gaussian == 0)));
precision = true_positive/(true_positive+false_positive);
recall = true_positive/(true_positive+false_negative);
f_score_test_given_gaussian = 2*(precision*recall)/(precision+recall);

fprintf('Using given SVM train and predict functions and the Gaussian kernel, we obtain: \nAccuracy: %0.4f \nF1 score: %0.4f \n',accuracy_test_given_gaussian,f_score_test_given_gaussian);

% use libsvm predict function
[f_score_validation_libSVM_gaussian_max_row, index_max_libSVM_gaussian_max_row] = max(f_score_validation_libSVM_gaussian);
[~,index_max_libSVM_gaussian_max_column] = max(f_score_validation_libSVM_gaussian_max_row);
predictions_test_libSVM_gaussian = svmpredict(test_set_labels,test_set_feature_vectors,model_array_libSVM_gaussian{index_max_libSVM_gaussian_max_row(index_max_libSVM_gaussian_max_column)}{index_max_libSVM_gaussian_max_column},'-q');
accuracy_test_libSVM_gaussian = mean(predictions_test_libSVM_gaussian == test_set_labels);
true_positive = sum(ismember(find(test_set_labels == 1),find(predictions_test_libSVM_gaussian == 1)));
false_positive = sum(ismember(find(test_set_labels == 0),find(predictions_test_libSVM_gaussian == 1)));
false_negative = sum(ismember(find(test_set_labels == 1),find(predictions_test_libSVM_gaussian == 0)));
true_negative = sum(ismember(find(test_set_labels == 0),find(predictions_test_libSVM_gaussian == 0)));
precision = true_positive/(true_positive+false_positive);
recall = true_positive/(true_positive+false_negative);
f_score_test_libSVM_gaussian = 2*(precision*recall)/(precision+recall);

fprintf('Using libSVM and the Gaussian kernel, we obtain: \nAccuracy: %0.4f \nF1 score: %0.4f \n',accuracy_test_libSVM_gaussian,f_score_test_libSVM_gaussian);