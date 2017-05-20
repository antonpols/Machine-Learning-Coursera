function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

    % Create New Figure
    figure; hold on;

        % ====================== YOUR CODE HERE ======================
        % Instructions: Plot the positive and negative examples on a
        %               2D plot, using the option 'k+' for the positive
        %               examples and 'ko' for the negative examples.
        %
        % =========================================================================
    
    feature1 = X(:,1);
    feature2 = X(:,2);
    
    plot(feature1(logical(y)),feature2(logical(y)),'ob')
    plot(feature1(logical(~y)),feature2(logical(~y)),'xr')
    
    legend('Admitted','Rejected');
    
    hold off;

end
