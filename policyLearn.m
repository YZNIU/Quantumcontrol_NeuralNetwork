function [net, perf] = policyLearn(data, inp)

target = data.target;
action = data.action;

% select training 'trainbr' as a training function
trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.

% generate an optimal control network based on fitting network structure
hiddenLayerSize = [10,6,8];
net = fitnet(hiddenLayerSize,trainFcn);

% input and output pre/post-processing functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% automatic division of data to training, validation, testing
net.divideFcn = 'dividerand';  % divide data randomly
net.divideMode = 'sample';  % divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% define performance function
net.performFcn = 'mse';  % Mean Squared Error
% need to add here a cardinality preference ... #########

% plot functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotregression', 'plotfit'};

% train network
[net,tr] = train(net,target,action);

% test network
y = net(target);
e = gsubtract(action,y);
perf = perform(net,action,y)

% recalculate training, validation and test performance
trainTargets = action .* tr.trainMask{1};
valTargets = action .* tr.valMask{1};
testTargets = action .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% view network
view(net)

% plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(action,y)
%figure, plotfit(net,target,action)

% deployment
% change (false) values to (true) to enable the following code blocks
if (false)
    % generate MATLAB function for neural network for application deployment in MATLAB 
    % scripts or with MATLAB Compiler and Builder tools, or simply to examine the calculations the trained neural network performs
    genFunction(net,'policyLearn');
    y = policyLearn(target);
end
if (false)
    % generate a matrix-only MATLAB function for neural network code generation with MATLAB Coder tools
    genFunction(net,'policyLearn','MatrixOnly','yes');
    y = myNeuralNetworkFunction(target);
end
if (false)
    % generate a Simulink diagram for simulation or deployment with Simulink Coder tools
    gensim(net);
end

