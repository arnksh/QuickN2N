function Data = min_max_norm(Data)
min_of_all = min(min(Data));
max_of_all = max(max(Data));
Data = (Data - min_of_all*(ones(size(Data))))./(max_of_all - min_of_all);

end
