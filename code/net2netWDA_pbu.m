clear all; clc
addpath ./ReqFunc/;
%%
para.nh = [70, 30, 20];        % cell-1 for IMS data and cell-2 for air compressor dat
para.epochs_DNN =500;          % training epochs
para.epochs_fineTune = 100;
para.mini_batch_size = 10;     % mini bacth size
para.sparsityParam = 0.1;
para.lambda = 0.05;            % weight decay parameter, regularization
para.beta = 0.8;               % k-l divergence coefficient
para.eta=0.9;
options.Method = 'lbfgs/';    %options for pre train AEs
options.maxIter =4;
options.display = 'on';
kthLayer=2; %The layer to be factorized to make deeper
add_nodes={[2,20]}; %first entry is the layer no and 2nd the no of nodes to be added.

%%
dataFolder = 'PBU_400';
tarData = {'tarData_1','tarData_2','tarData_3','tarData_4','tarData_5'};
fileID = fopen('./logs_PBU/net2netWDA_40.txt','w');

row = 34; %34 for PBU_400 AND 44 for PBU_40
%% Network transformation and fine tune
for ca=3:4
    fprintf(fileID, '\n\n*********Tar%d**********\n',ca);

    for i = 1:4
        load(['../' dataFolder '/tar' int2str(ca) '/' tarData{i}], 'Y');
        load('./logs_PBU/TeacherNet.mat')
        
        % First make the network deeper and then wider as required
        ipSize = size(Y.training_inputs,2); 
        opSize = size(unique(Y.training_results),1);
        net = Net2NetDeeper(net.W, net.b, net.nh, kthLayer, ipSize, opSize); %make the network deeper by one step
        net = Net2NetWider(net.W, net.b, net.nh, add_nodes, ipSize, opSize);
        
        % Fine tune the new network FOR DOAMIN ADAPTATION
        net = fineTuneWDA_1(net, Y, tarData{i}, para);
        
        % Validate the network
        acc = TestNetwork(Y.test_inputs', Y.test_results', net);
        fprintf('Testing Accuracy for %s: %% acc =  %3.1f\n', tarData{i}, mean(acc));

        fprintf(fileID, 'Testing Accuracy for %s for student model: %% acc =  %3.1f\n', tarData{i}, mean(acc));
        fprintf(fileID,'=============================================\n\n');
        xrange = sprintf('J%d', row);
        xlswrite('resultN2N.xlsx',mean(acc),1,xrange)
        row =row+1;
        % save(['./logs_net2net/studentNet_' tarData{i} '.mat'], 'net')
    end
end
fclose(fileID);

%% High Level Data Generation for SVM, RF classification


% test_inputs =(feedforward1(InputData.test_inputs',net.W,net.b,l-1))';
% training_inputs=(feedforward1(InputData.training_inputs',net.W,net.b,l-1))';
% training_results = InputData.training_results;
% test_results = InputData.test_results;
% save(['../' dataFolder '/logs/high_' nameOfData1 '.mat'],'training_inputs', 'training_results','test_inputs','test_results')
% save(['../' dataFolder '/logs/studentNet' nameOfData '.mat'], 'netStudent')
% fclose(fileID);
