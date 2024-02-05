function y = vectorizeData(results,o) 
    y = zeros(o,length(results));

    for i = 1:length(results)
        y(results(i)+1,i) = 1.0;
    end
end