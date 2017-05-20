function word_indices = processEmail_to_indices(processed_email_contents,vocabulary)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%

    words_email = textscan(processed_email_contents,'%s');
    words_email = words_email{:};
      
    word_indices = zeros(numel(words_email),1);
    indices_to_remove = [];
    for i = 1:numel(words_email)
        word_index = find(ismember(vocabulary,words_email{i}),1);
        if ~isempty(word_index)
            word_indices(i) = word_index;
        else
            indices_to_remove = [indices_to_remove i];
        end      
    end  
    word_indices(indices_to_remove) = [];

end