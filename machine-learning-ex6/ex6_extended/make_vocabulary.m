function vocabulary = make_vocabulary(processed_email_contents,count_words)
    
    p = gcp('nocreate');
    if isempty(p)
        parpool();
    end
    
    words_email = cell(numel(processed_email_contents),1);
    parfor_progress(numel(processed_email_contents));
    parfor i = 1:numel(processed_email_contents)    
        words_email(i) = textscan(processed_email_contents{i},'%s');
        parfor_progress;
    end
    parfor_progress(0);
    
    [unique_words_email,~,indices_words_email_unique]= unique(cat(1,words_email{:}));
    times_unqiue_words_email = hist(indices_words_email_unique,numel(unique_words_email));    
    if exist('count_words','var')
        vocabulary = unique_words_email(times_unqiue_words_email>count_words);
    else
        vocabulary = unique_words_email(times_unqiue_words_email>100);
    end
end