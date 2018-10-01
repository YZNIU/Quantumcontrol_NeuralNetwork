function [net, perf, xi, ai] = envLearn(data, inp)

% extract data
X = data.X;
T = data.T;
% Tnf = data.Tnf;

if isempty(inp.net) % in case this is the first time the network is formed, otehrwise boot strap the previous net
    if inp.ML
        delay =2;
        net = feedforwardnet([20 10 10]);
        net.layerConnect = [0 0 0 1;
            1 0 0 0;
            0 1 0 0;
            0 0 1 0];
        net.outputs{4}.feedbackMode = 'closed';
        net.outputs{4}.feedbackMode = 'open';
        net.inputWeights{1,1}.delays = 1:delay;
        net.inputWeights{1,2}.delays = 1:delay;
        net.inputs{1}.size = size(X{1},1);
        view(net)
        
        net.layers{1}.transferFcn = 'tansig'; % radbas
        net.layers{2}.transferFcn = 'tansig';
        net.layers{3}.transferFcn = 'purelin';
        
        net.plotFcns = {'plotperform' ,'plottrainstate',...
            'ploterrhist','plotregression','plotresponse'};
    else
        % net = narxnet(0:inp.NARXdelay,1:inp.NARXdelay,12);
        net = narxnet(1:2,1:2,25);
        
        transFun = '''tansig''';%'''purelin''';%'''myTransFun'''; %'''hardlims'''; % '''satlins'''; %% %  '''tansig''';%  %
        eval(['net.layers{1}.transferFcn = ', transFun, ';']);
    end
    net.divideFcn =  'divideblock'; %'divideint'; %  'dividerand'; 'divideblock';
    net.trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.
    net.trainParam.max_fail = 30;
    net.trainParam.min_grad = 1e-6;
    net.trainParam.time = 60*360; % limited time for training
    net.performParam.normalization = 'standard';
    
else
    net = inp.net;
    net.trainParam.time = 60*2; % limited time for training
    [net,xi, ai] = openloop(net,inp.xi, inp.ai);
end

[x,xi,ai,t] = preparets(net,X,{},T);
% [x,xi,ai,t] = preparets(net,X,Tnf,T);

net = train(net,x,t,xi);
% net = train(net,x,t,xi,ai);

y = net(x,xi,ai);
perf = gsubtract(t,y);
view(net)


% % %-----------------------------------------------------------------------------------------------
% % % %-----------------------------------------------------------------------------------------------
% net = closeloop(net);
% view(net)
% [Xs,Xi,Ai] = preparets(net,X,{},T);
% y = net(Xs,Xi,Ai);


% % %-----------------------------------------------------------------------------------------------
% % %-----------------------------------------------------------------------------------------------
% %
% % multiLayer = true;
% %
% % if multiLayer
% %     net = feedforwardnet([32 36 17]);
% %     net.layerConnect = [0 0 0 1; 1 0 0 0;0 1 0 0; 0 0 1 0];
% % else
% %     net = feedforwardnet([24]);
% %     net.layerConnect = [0 1; 1 0 ];
% % end
% %
% % delay = 24;
% % if multiLayer
% %     net.outputs{4}.feedbackMode = 'closed';
% %     net.outputs{4}.feedbackMode = 'open';
% % else
% %     net.outputs{2}.feedbackMode = 'closed';
% %     net.outputs{2}.feedbackMode = 'open';
% % end
% %
% % % in principle should be from 0, but then need to extent an initial
% % % condition
% % net.inputWeights{1,1}.delays = 1:delay;%1:delay;
% %
% % net.inputWeights{1,2}.delays = 1:delay;
% % net.inputs{1}.size = size(XX{1},1);
% % net.inputs{2}.size = size(TT{1},1);
% %
% %
% % [Xs,Xi,Ai,Ts] = preparets(net,XX,{},TT);
% %
% % view(net)
% %
% % for ii=1:size(net.IW,2)
% %     net.IW{ii} = rand(size(net.IW{ii}));
% % end
% %
% % net = train(net,Xs,Ts,Xi,Ai);
% % Y = net(Xs,Xi,Ai);
% %
% % perf = perform(net,Ts,Y)
% %
% %
% % %----------------------------------------------------------
% %
% % netc = closeloop(net);
% % netc.name = [net.name ' - Closed Loop'];
% %
% % [inputs,inputStates,layerStates,targets] = preparets(netc,inputSeriesVal,{},targetSeriesVal);
% %
% % yPred = netc(inputs,inputStates,layerStates);
% %
% % perf = perform(net,yPred,targetSeriesVal(delay+1:end));

%------------------------------------------------------
%-----------------------------------------------------------------------------------------------
%-----------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         net = feedforwardnet([20 12 15]);
        %         net.layerConnect = [0 0 0 1;
        %             1 0 0 0;
        %             0 1 0 0;
        %             0 0 1 0];
        %          net.outputs{4}.feedbackMode = 'closed';
        %          net.outputs{4}.feedbackMode = 'open';
        %         net.inputWeights{1}.delays = 1:2; % 0:inp.NARXdelay;
        %         net.layerWeights{1,4}.delays = 1:2; % 1:inp.NARXdelay;
        %         net.inputs{1}.size = size(X{1},1);

% T[X,t] = simplenarx_dataset;
% net = narxnet(1:2,1:2,20);
% [Xs,Xi,Ai,Ts] = preparets(net,X,{},T);
% net = train(net,Xs,Ts,Xi,Ai);
% view(net)
% y = net(Xs,Xi,Ai);
% perf = perform(net,Y,T)
%
%
% net = closeloop(net);
% view(net)
% [Xs,Xi,Ai] = preparets(net,X,{},T);
% y = net(Xs,Xi,Ai);