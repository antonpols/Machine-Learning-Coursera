clear;
close all;
p = gcp('nocreate');
if isempty(p)
    parpool();
end

%% load normal mails

dd_normal = dir(strcat(pwd,'/email/normal/'));
dd_normal = dd_normal(~cat(1,dd_normal.isdir)); %remove directory entries

file_names_normal = {dd_normal.name};
folder_normal = {dd_normal.folder};

raw_text_data_normal = cell(numel(file_names_normal),1);
parfor_progress(numel(file_names_normal));
parfor i = 1:numel(file_names_normal)
    raw_text_data_normal{i} = fileread(strcat(folder_normal{i},'/',file_names_normal{i}));
    parfor_progress;
end
parfor_progress(0);

%% load spam mails

dd_spam = dir(strcat(pwd,'/email/spam/'));
dd_spam = dd_spam(~cat(1,dd_spam.isdir)); %remove directory entries

file_names_spam = {dd_spam.name};
folder_spam = {dd_spam.folder};

raw_text_data_spam = cell(numel(file_names_spam),1);
parfor_progress(numel(raw_text_data_spam));
parfor i = 1:numel(raw_text_data_spam)
    raw_text_data_spam{i} = fileread(strcat(folder_spam{i},'/',file_names_spam{i}));
    parfor_progress;
end
parfor_progress(0);

%% process each email

parfor_progress(numel(raw_text_data_normal));
processed_normal = cell(numel(raw_text_data_normal),1);
indices_to_remove = [];
parfor i = 1:numel(raw_text_data_normal)
    processed_normal_temp = processEmail(raw_text_data_normal{i});
    if ~isempty(processed_normal_temp)
        processed_normal{i} = processed_normal_temp;
    else
        indices_to_remove = [indices_to_remove i];
    end
    parfor_progress;
end
parfor_progress(0);
processed_normal(indices_to_remove) = [];

parfor_progress(numel(raw_text_data_spam));
processed_spam = cell(numel(raw_text_data_spam),1);
indices_to_remove = [];
parfor i = 1:numel(raw_text_data_spam)
    processed_spam_temp = processEmail(raw_text_data_spam{i});
    if ~isempty(processed_spam_temp)
        processed_spam{i} = processed_spam_temp;
    else
        indices_to_remove = [indices_to_remove i];
    end
    parfor_progress;
end
parfor_progress(0);
processed_spam(indices_to_remove) = [];

%% make vocabulary

vocabulary = make_vocabulary([processed_normal; processed_spam],100);

%% construct feature vectors

parfor_progress(numel(processed_normal));
word_indices_normal = cell(numel(processed_normal),1);
feature_vectors_normal = cell(numel(processed_normal),1);
parfor i = 1:numel(processed_normal)
    word_indices_normal{i} = processEmail_to_indices(processed_normal{i},vocabulary);
    feature_vectors_normal{i} = emailFeatures(word_indices_normal{i},vocabulary);
    parfor_progress;
end
parfor_progress(0);

parfor_progress(numel(processed_spam));
word_indices_spam = cell(numel(processed_spam),1);
feature_vectors_spam = cell(numel(processed_spam),1);
parfor i = 1:numel(processed_spam)
    word_indices_spam{i} = processEmail_to_indices(processed_spam{i},vocabulary);
    feature_vectors_spam{i} = emailFeatures(word_indices_spam{i},vocabulary);
    parfor_progress;
end
parfor_progress(0);

%% make the training, cross validation and test sets
total_samples = numel(processed_normal) + numel(processed_spam);
indices_random_sampled = randsample(total_samples,total_samples);

ratio_train_set = 0.6;
ratio_validation_set = 0.3;
ratio_test_set = 0.1;

total_dataset_feature_vectors = [cat(2,feature_vectors_normal{:}) cat(2,feature_vectors_spam{:})]';
total_dataset_labels = [zeros(numel(feature_vectors_normal),1); ones(numel(feature_vectors_spam),1)];

train_set_feature_vectors = total_dataset_feature_vectors(indices_random_sampled(1:floor(numel(indices_random_sampled)*ratio_train_set)),:);
train_set_labels = total_dataset_labels(indices_random_sampled(1:floor(numel(indices_random_sampled)*ratio_train_set)),:);

validation_set_feature_vectors = total_dataset_feature_vectors(indices_random_sampled(floor(numel(indices_random_sampled)*ratio_train_set)+1:floor(numel(indices_random_sampled)*(ratio_train_set+ratio_validation_set))),:);
validation_set_labels = total_dataset_labels(indices_random_sampled(floor(numel(indices_random_sampled)*ratio_train_set)+1:floor(numel(indices_random_sampled)*(ratio_train_set+ratio_validation_set))),:);

test_set_feature_vectors = total_dataset_feature_vectors(indices_random_sampled(floor(numel(indices_random_sampled)*(ratio_train_set+ratio_validation_set))+1:numel(indices_random_sampled)),:);
test_set_labels = total_dataset_labels(indices_random_sampled(floor(numel(indices_random_sampled)*(ratio_train_set+ratio_validation_set))+1:numel(indices_random_sampled)),:);

%% save the results

save('preprocessed_total_dataset.mat','train_set_feature_vectors', 'train_set_labels','validation_set_feature_vectors','validation_set_labels','test_set_feature_vectors','test_set_labels');