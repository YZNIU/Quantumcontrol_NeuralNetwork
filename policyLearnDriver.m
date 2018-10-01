% load data
inp = [];

% load('D:\Box\Box Sync\MIT-IBM Watson AI Lab - Projects\AI4Q\data\BlochRL\supervisedPolicyLearn.mat')
load('D:\Box\Box Sync\MIT-IBM Watson AI Lab - Projects\AI4Q\data\BlochRL\target_action_data_11pmApril2nd.mat')

data.target = target';
data.action = action;

% train network
[net, perf] = policyLearn(data, inp);


%-----------------------------------------------------------------------------------------------
% load unseen data
load('D:\Box\Box Sync\MIT-IBM Watson AI Lab - Projects\AI4Q\data\BlochRL\target_action_validate_12amApril3rd.mat')

data.targetTest = target';
data.actionTest = action;

%-----------------------------------------------------------------------------------------------
% simulate network
y = net(data.targetTest);
e = gsubtract(data.actionTest,y);
perf = perform(net,data.actionTest,y)
