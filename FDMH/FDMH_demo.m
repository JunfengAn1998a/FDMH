close all; clear all; clc; clear memory;

%% Config
db_name = 'wiki';
nbits = 16;
runtimes = 3;

%% Load dataset
fprintf('\nLoading dataset...\n');
dataset_path = ['./data/',db_name,'_cnn.mat'];
load(dataset_path);
if strcmp(db_name, 'wiki')
    X{1} = [I_tr;I_te];
    X{2} = [T_tr;T_te];
    class_n = max(L_tr);
    onehot_tr = full(ind2vec(L_tr',class_n));
    onehot_te = full(ind2vec(L_te',class_n));
    gnd  = [onehot_tr';onehot_te'];
else
    X{1} = [I_db;I_te];
    X{2} = [T_db;T_te];
    gnd  = [L_db;L_te];
end

%% Set parameters
pars.n_anchor = 1000;
pars.M = size(X,2);
pars.bits = nbits;
pars.lambda = 1e-1;
pars.beta = 1e-6;
pars.rho = 5e5;
pars.rho0 = 1e6;
pars.miu = 1.6;
if strcmp(db_name, 'wiki')
    pars.theta = 1e6;
    pars.eta = 9e10;
    pars.rho = 3e2;
    pars.lambda =1e0;
    pars.beta = 1e0;
elseif strcmp(db_name, 'mir')
    pars.theta = 3e4;
    pars.eta = 7e11;
    pars.rho = 5e5;
    pars.lambda =1e2;
elseif strcmp(db_name, 'coco2017')
    pars.theta = 9e5;
    pars.eta = 3e13;
    pars.rho = 1e-1;
elseif strcmp(db_name, 'nus')
    pars.theta = 2e6;
    pars.eta = 2e12;
end


%% Anchor graph
fprintf('\nComputing anchor graphs...\n');
n_Sam = size(X{1},1);
anchor_indexes = randsample(n_Sam, pars.n_anchor);
for it = 1:pars.M
    X{it} = double(X{it});
    anchor = X{it}(anchor_indexes,:);
    Dis = EuDist2(X{it},anchor,0);
    sigma = mean(mean(Dis)).^0.5;
    feavec = exp(-Dis/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feavec', mean(feavec',2));
end

%% Separate train and test index
tt_num = size(I_te,1);
data.gnd = gnd;
tt_idx = n_Sam-tt_num+1:n_Sam;
tr_idx = 1:n_Sam-tt_num;
ttgnd = gnd(tt_idx,:);
trgnd = gnd(tr_idx,:);

%% Groungd truth
WtrueTestTraining = ttgnd * trgnd'>0;
clear gnd trgnd ttgnd;

data.indexTrain = tr_idx;
data.indexTest = tt_idx;
ttfea = cell(1,pars.M);
for i = 1:pars.M
    data.X{i} = normEqualVariance(X{i}')';
    ttfea{i} = data.X{i}(:,tt_idx);
end
clear X;

fprintf('\nStart running...\n');
mMAP = zeros(1,runtimes);
for t = 1:runtimes
    pars.n_iters = t;
    [B_trn,alpha,P,R,Q] = model(pars,data);

    %% Generate test hash codes
    B_tst = hashFunction(ttfea,alpha,P,R);

    %% Evaluation
    B1 = compactbit(B_trn');
    B2 = compactbit(B_tst');
    DHamm = hammingDist(B2, B1);
    [~, orderH] = sort(DHamm, 2);
    mMAP(t) = calcMAP(orderH, WtrueTestTraining);
    fprintf('The %d-th runtime, mAP: %.4f\n', t, mMAP(t));
end
fprintf('Finished\n');
mean_map = mean(mMAP);
fprintf('The mean mAP of %d runtimes: %.4f\n', runtimes, mean_map);

