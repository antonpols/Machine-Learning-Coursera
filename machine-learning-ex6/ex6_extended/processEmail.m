function processed_email = processEmail(email_contents)
%PROCESSEMAIL preprocesses a the body of an email

    % ========================== Preprocess Email ===========================

    % Find the Headers ( \n\n and remove )
    % Uncomment the following lines if you are working with raw emails with the
    % full headers

    hdrstart_linux = strfind(email_contents, ([char(10) char(10)]));
    hdrstart_windows = strfind(email_contents, ([char(13) char(10) char(13) char(10)]));
    hdrstart = sort([hdrstart_linux hdrstart_windows]);

    email_contents = email_contents(hdrstart(1):end);

    % Lower case
    email_contents = lower(email_contents);

    % Strip all HTML
    % Looks for any expression that starts with < and ends with > and replace
    % and does not have any < or > in the tag it with a space
    email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

    % Handle Numbers
    % Look for one or more characters between 0-9
    email_contents = regexprep(email_contents, '[0-9]+', 'number');

    % Handle URLS
    % Look for strings starting with http:// or https://
    email_contents = regexprep(email_contents, ...
                               '(http|https)://[^\s]*', 'httpaddr');

    % Handle Email Addresses
    % Look for strings with @ in the middle
    email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

    % Handle $ sign
    email_contents = regexprep(email_contents, '[$]+', 'dollar');


    % ========================== Tokenize Email ===========================
    
    processed_email = '';
    
    while ~isempty(email_contents)

        % Tokenize and also get rid of any punctuation
        [str, email_contents] = ...
           strtok(email_contents, ...
                  [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);

        % Remove any non alphanumeric characters
        str = regexprep(str, '[^a-zA-Z0-9]', '');

        % Stem the word 
        % (the porterStemmer sometimes has issues, so we use a try catch block)
        try str = porterStemmer(strtrim(str)); 
        catch str = ''; continue;
        end;

        % Skip the word if it is too short
        if length(str) < 1
           continue;
        end
        
        processed_email = [processed_email,' ',str];
    end
   
end
