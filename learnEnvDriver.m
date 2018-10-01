% environment training driver

clear all;
close all;
dbstop if error
%----------------------------------------------------------------------------
% choose simulation model
inp.NARXdelay = 1; %  delays for narx model
inp.ML = true; % multi-layer vs. single level neural network
inp.derivatives = false; % model computes derivatives

%----------------------------------------------------------------------------
% form NARX 
inp.net = [];

%----------------------------------------------------------------------------
%select file to load
disp ('loading data')

[fn, pn] = uigetfile('*.mat');
ffn = [pn, fn];
load(ffn)
%---------------------------------------------------------------------------
disp ('format time series data')

for ii=1:size(action,3)    
    data.X{ii} = squeeze(action(:,:,ii));
end

for ii=1:size(uhist,3)-1
    data.T{ii} = squeeze(uhist(:,:,ii)); 
end

% for ii=1:size(uhist,3)
%     data.T{ii} = squeeze(uhist(:,:,ii)); 
% end

% catsamples(x1,x2,...,xn)
% data.Tnf = con2seq(unow');

%-----------------------------------------------------------------------------------------------------
% learn environment
disp ('learn environment')
[inp.net, perf, inp.xi, inp.ai] = envLearn(data, inp);
% close narx loop
inp.net = closeloop(inp.net);
view(inp.net)

%-----------------------------------------------------------------------------------------------
% load unseen data
disp ('load data to be simlulated (unseen)')

[fn, pn] = uigetfile('*.mat');
ffn = [pn, fn];
load(ffn)

%load('D:\Box\Box Sync\MIT-IBM Watson AI Lab - Projects\AI4Q\data\BlochRL\dataApril2nd415pm.mat')
%-----------------------------------------------------------------------------------------------
disp ('format time series data')

for ii=1:size(action,3)    
    data.X{ii} = squeeze(action(:,:,ii));
end

for ii=1:size(uhist,3)-1
    data.T{ii} = squeeze(uhist(:,:,ii)); 
end

%-----------------------------------------------------------------------------------------------
% simulate network
[Xs,Xi,Ai] = preparets(inp.net,data.Xts,{},data.Tts);
y = inp.net(Xs,Xi,Ai);

% load('D:\Box\Box Sync\MIT-IBM Watson AI Lab - Projects\AI4Q\data\BlochRL\BlochNRAXTraining20data.mat')
% data.X = con2seq(action');
% data.T = con2seq(unext');

% pathName = "D:\Box\Box Sync\MIT-IBM Watson AI Lab - Projects\AI4Q\data\BlochRL\";
% fn = 'dataApril2nd4pm.mat';
% fn = 'narxDATA_April2nd11pm.mat';
% fn = 'DarxApril3rd12am.mat';
% fn = 'oneSampleApril3rd2pm.mat';
% fn = 'DIFFactionNarxApri3rd3PM.mat';
% fn = 'smallAction_200sample1000Time_april3_4pm.mat';
% fn = '100sample10000TimestepDiffaction04032018.mat';

% fn = strcat(pathName, fn);
% load (fn)

%----------------------------------------------------------------------------
% % optimization settings
% if inp.derivatives
%     opts = optimoptions(@fmincon,'Algorithm','interior-point','Display','iter-detailed', ...
%         'TolFun',1e-4,'UseParallel',0, 'ScaleProblem','obj-and-constr', ...
%         'TolX', 1e-4,'GradObj','off','GradConstr','on');%, 'DerivativeCheck','on'); % 'GradObj','off'
% else
%     opts = optimoptions(@fmincon,'Algorithm','interior-point','Display','iter-detailed', ...
%         'TolFun',1e-4,'UseParallel',0, 'ScaleProblem','obj-and-constr', ...
%         'FinDiffRelStep',1e-8,'TolX', 1e-4); % 'GradObj','off'
% end
