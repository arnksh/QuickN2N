
function [stack] = form_stack(sae1OptTheta,sae2OptTheta,sae3OptTheta,inputSize,hiddenSizeL1,hiddenSizeL2,hiddenSizeL3)


        stack = cell(3,1);
        stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
            hiddenSizeL1, inputSize);
        stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
        
        stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
            hiddenSizeL2, hiddenSizeL1);
        stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
        
        stack{3}.w = reshape(sae3OptTheta(1:hiddenSizeL3*hiddenSizeL2), ...
            hiddenSizeL3, hiddenSizeL2);
        stack{3}.b = sae3OptTheta(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);


end
