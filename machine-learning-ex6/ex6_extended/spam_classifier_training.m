clear;
close all;
p = gcp('nocreate');
if isempty(p)
    parpool();
end

%% load the preprocced data

load('preprocessed_total_dataset.mat');

%% train the SVM

%linear kernel SVM
C = [10^-3 3*10^-3 10^-2 3*10^-2 10^-1 3*10^-1 10^0 3*10^0 10^1 3*10^1 10^2 3*10^2 10^3]; 

% use given SVM train function
model_array_given_linear = cell(1,numel(C));
parfor_progress(numel(C));
parfor i = 1:numel(C)   
    model_array_given_linear{i} = svmTrain(train_set_feature_vectors, train_set_labels, C(i), @linearKernel);
    parfor_progress;    
end
parfor_progress(0);

% use libsvm train function
model_array_libSVM_linear = cell(1,numel(C));
parfor_progress(numel(C));
parfor i = 1:numel(C)   
    model_array_libSVM_linear{i} = svmtrain(train_set_labels, train_set_feature_vectors, sprintf('-q -t 0 -c %f',C(i)));  
    parfor_progress;
end
parfor_progress(0);


%gaussian kernel SVM
C = [10^0 3*10^0 10^1 3*10^1]; 
sigma = [10^0 3*10^0 10^1 3*10^1 10^2 3*10^2 10^3 3*10^3 10^4 3*10^4 10^5];

% use given SVM train function
model_array_given_gaussian = cell(1,numel(C));
parfor_progress(numel(C));
parfor i = 1:numel(C)
    model_array_given_gaussian_temp = cell(1,numel(sigma));
    for j = 1:numel(sigma)
        model_array_given_gaussian_temp{j} = svmTrain(train_set_feature_vectors, train_set_labels, C(i),  @(x, l) gaussianKernel(x, l, sigma(j)));         
    end
    model_array_given_gaussian{i} = model_array_given_gaussian_temp;
    parfor_progress;
end
parfor_progress(0);

% use libsvm train function
model_array_libSVM_gaussian = cell(1,numel(C));
parfor_progress(numel(C));
parfor i = 1:numel(C)   
    model_array_libSVM_gaussian_temp = cell(1,numel(sigma));
    for j = 1:numel(sigma)
        model_array_libSVM_gaussian_temp{j} = svmtrain(train_set_labels, train_set_feature_vectors, sprintf('-q -t 2 -c %f -g %f',C(i),1/(2*sigma(j)^2)));         
    end
    model_array_libSVM_gaussian{i} = model_array_libSVM_gaussian_temp;
    parfor_progress;
end
parfor_progress(0);

%% save the results

save('trained_models.mat','model_array_given_linear','model_array_given_gaussian','model_array_libSVM_linear','model_array_libSVM_gaussian','C','sigma','-v7.3');