clear all; clc
addpath ./ReqFunc/;                addpath ('./ReqFunc/softmax')
addpath ./ReqFunc/minFunc/;        addpath ('./ReqFunc/starter')
addpath ('./ReqFunc/stl_exercise')
%%
para.adapt  = 0; %Choose 0 for DA and -1 for WDA
para.nh = [70, 30, 20];   
para.epochs_DNN =500;          
para.epochs_fineTune = 100;
para.mini_batch_size =10;     
para.sparsityParam = 0.1;   
para.lambda = 0.05;         
para.beta = 0.8;            
para.eta=1.5;
options.Method = 'lbfgs';   
options.maxIter =4;
options.display = 'on';
kthLayer=2;          %The layer to be factorized to make deeper
add_nodes={[2,20]};  %first entry is the layer no and 2nd the no of nodes to be added.
%%
dataFolder = 'CWRU_400_1';
srcData  = 'src_7_0';
tarData = {'FE_tar_7_1', 'FE_tar_7_2', 'FE_tar_7_3', 'FE_tar_14_1', 'FE_tar_14_2',....
         'FE_tar_14_3', 'FE_tar_21_1', 'FE_tar_21_2', 'FE_tar_21_3', 'ims_tar'};
fileID = fopen('./logs_CWRU/net2net_400_ims_wda.txt','w');

row = 6; %6 for CWRU_400 AND 17 for CWRU_40
% row = 15; %15 for CWRU_400 AND 26 for CWRU_40
%% Network transformation and fine tune
for i = 1:10
    load(['../' dataFolder '/' tarData{i}], 'Y');
    load('.\logs_CWRU\TeacherNet.mat')
    
    % First make the network deeper and then wider as required
    ipSize = size(Y.training_inputs,2); 
    opSize = size(unique(Y.training_results),1);
    net = Net2NetDeeper(net.W,net.b, net.nh, kthLayer, ipSize, opSize); %make the network deeper by one step
    net = Net2NetWider(net.W,net.b, net.nh, add_nodes, ipSize, opSize);

    % Fine tune the new network with domain Adaptation
    net = fineTuneDA_1(net, Y, tarData{i}, para);
    
    % Validate the network
    
    
    acc = TestNetwork(Y.test_inputs',Y.test_results', net);
    
    fprintf('Testing Accuracy for %s: %% acc =  %3.1f\n', tarData{i}, mean(acc));
    
    xrange = sprintf('J%d',row);
    xlswrite('resultN2N.xlsx',mean(acc),1,xrange)
    row =row+1;
    save(['./logs_CWRU/studentNet_' tarData{i} '.mat'], 'net')

    fprintf(fileID, 'Testing Accuracy for %s for student model: %% acc =  %3.1f\n', tarData{i}, mean(acc));
    fprintf( fileID,'=============================================\n\n');
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
