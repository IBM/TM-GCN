%% Read data and compute output
addpath(genpath('tensor_toolbox_2.6'))
% Settings
edge_life = 10;
no_diag = 20;
dataset = 'Reddit';
make_symmetric = true; % Remember: symmetric generally performs WORSE
M_choice = 2;
M_rownorm =0;
disp(dataset)

% Dataset dependent settings
if strcmp(dataset, 'Bitcoin OTC')
    data = csvread('data/Bitcoin_OTC/soc-sign-bitcoinotc.csv');
    save_file_location = 'data/Bitcoin_OTC/';
    save_file_name = 'saved_content_bitcoin_otc.mat';
    time_delta = 60*60*24*14; % 2 weeks
    no_train_samples = 95;
    no_val_samples = 20;
    no_test_samples = 20;
elseif strcmp(dataset, 'Bitcoin Alpha')
    data = csvread('data/Bitcoin_Alpha/soc-sign-bitcoinalpha.csv');
    save_file_location = 'data/Bitcoin_Alpha/';
    save_file_name = 'saved_content_bitcoin_alpha.mat';
    time_delta = 60*60*24*14; % 2 weeks
    no_train_samples = 95;
    no_val_samples = 20;
    no_test_samples = 20;
elseif strcmp(dataset, 'Reddit')
    data = textread('data/Reddit/soc-redditHyperlinks-body.tsv');
    save_file_location = 'data/Reddit/';
    save_file_name = 'saved_content_reddit.mat';
    time_delta = 60*60*24*7*2; % 2 weeks
    no_train_samples = 66;
    no_val_samples = 10;
    no_test_samples = 10;
elseif strcmp(dataset, 'Chess')
    data = csvread('data/chess/out.chess');
    save_file_location = 'data/chess/';
    save_file_name = 'saved_content_chess.mat';
    time_delta = 60*60*24*31; % 31 days
    no_train_samples = 80;
    no_val_samples = 10;
    no_test_samples = 10;
elseif strcmp(dataset, 'hep-th')
    data = readmatrix('data/hep-th/out.ca-cit-HepTh.csv', 'NumHeaderLines', 2);
    save_file_location = 'data/hep-th/';
    save_file_name = 'saved_content_hep-th.mat';
    time_delta = 60*60*24*60; % 60 days
    no_train_samples = 155;
    no_val_samples = 20;
    no_test_samples = 20;    
elseif strcmp(dataset, 'wikiconflict')
    data = readmatrix('data/wikiconflict/out.wikiconflict.csv', 'NumHeaderLines', 2);
    save_file_location = 'data/wikiconflict/';
    save_file_name = ['saved_content_wikiconflict_M', num2str(no_diag), '.mat'];
    time_delta = 60*60*24*31; % 31 days
    no_train_samples = 69;
    no_val_samples = 10;
    no_test_samples = 10;  
elseif strcmp(dataset, 'amlsim')
    data_temp = readmatrix('data/amlsim/1Kvertices-100Kedges/transactions.csv');
    data = [data_temp(:,2)+1, data_temp(:,3)+1, data_temp(:,7) data_temp(:,6)+1]; % extract sender, receiver, fraud flag, timestamp
    save_file_location = 'data/amlsim/1Kvertices-100Kedges/';
    save_file_name = ['saved_content_amlsim', num2str(no_diag), '.mat'];
    time_delta = 1;
    no_train_samples = 150;
    no_val_samples = 25;
    no_test_samples = 25;
elseif strcmp(dataset, 'uci')
    data_temp = readtable('data/uci/OCnodeslinks.txt');
    save_file_location = 'data/uci/';
    save_file_name = 'saved_content_uci.mat';
    data = zeros(size(data_temp));
    data(:,2:4) = data_temp{:,2:4};
    data(:,1) = datenum(data_temp{:,1});
    no_train_samples = 62;
    no_val_samples = 13;
    no_test_samples = 13;
    time_delta = 1;
    tot = no_train_samples + no_val_samples + no_test_samples;
    t_min = min(data(:,1));
    t_max = max(data(:,1));
    data(:,1) = (data(:,1)-t_min)/(t_max-t_min);
    data(:,1) = floor(data(:,1)*(tot))+1;
    data(end,1) = 88;
    data = [data(:,2) data(:,3) data(:,4) data(:,1)]; % sender, receiver, text message weight (no. char), time stamp
elseif strcmp(dataset, 'eu-core')
    data_temp = readtable('data/eu-core/email-Eu-core-temporal.txt');
    save_file_location = 'data/eu-core/';
    save_file_name = 'saved_content_eu-core.mat';
    time_delta = 6*24*60*60; % Each slice contains 6 days
    no_train_samples = 93;
    no_val_samples = 20;
    no_test_samples = 20;
    data = zeros(size(data_temp, 1), 4);
    data(:, [1 2 4]) = data_temp{:, :};
    data(:, [1 2]) = data(:, [1 2]) + 1;
    data(:, 3) = 1;
    data(:, 4) = data(:, 4) + 1;
else
    error('Invalid dataset')
end

% Create full tensor
if strcmp(dataset, 'Chess')
    dates = unique(data(:,4));
    no_time_slices = length(dates);  
else
    no_time_slices = floor((max(data(:,4)) - min(data(:,4)))/time_delta);
end
N = max([data(:,1); data(:,2)]);
T = no_train_samples;
TT = no_time_slices;

% Create M
M = zeros(T);
for d = 1:no_diag
    if M_choice ==1
        M = M + diag(ones(T+1-d, 1), 1-d);
    elseif M_choice ==2
        M = M + diag(ones(T+1-d, 1)/d, 1-d);
    end
end
if M_rownorm ==1
        M = M./sum(abs(M),2);
end

% Create A and A_labels
if ~strcmp(dataset, 'Chess')
    data = data(data(:,4) < min(data(:,4)) + TT*time_delta, :);
    start_time = min(data(:,4));
end
tensor_idx = zeros(size(data, 1), 3);
tensor_val = ones(size(data, 1), 1);
tensor_labels = zeros(size(data, 1), 1);

for t = 1:TT
    if strcmp(dataset, 'Chess')
        idx = data(:, 4) == dates(t);
    else
        end_time = start_time + time_delta;
        idx = (data(:, 4) >= start_time) & (data(:, 4) < end_time);
        start_time = end_time;
    end
    tensor_idx(idx, 2:3) = data(idx, 1:2);
    tensor_idx(idx, 1) = t;
    tensor_labels(idx) = data(idx, 3);
end

A = sptensor(tensor_idx, tensor_val, [TT, N, N]);
A_labels = sptensor(tensor_idx, tensor_labels, [TT, N, N]);

if strcmp(dataset, 'wikiconflict') 
    % Reduce size of dataset
    A = sptensor(A_labels.subs, A(A_labels.subs), [TT, N, N]);
    Atemp = sparse(N, N);
    for k = 1:size(A,1)
        Atemp = Atemp + spmatrix(A(k,:,:));
    end
    idx_sum = sum(Atemp, 1);
    idx_keep = idx_sum >= 100;
    idx_keep = sparse(1:N, 1:N, idx_keep)*(1:N).';
    idx_keep = idx_keep(idx_keep > 0);
    A = A(:, idx_keep, idx_keep);
    A_labels = A_labels(:, idx_keep, idx_keep);
    tensor_idx = A_labels.subs;
    tensor_labels = A_labels.vals;
    N = length(idx_keep);
end

% Make symmetric
if make_symmetric
    B = sptensor(size(A));
    for k = 1:size(B, 1)
        temp = spmatrix(A(k,:,:)) + spmatrix(A(k,:,:)).';
        B(k,:,:) = sptensor(temp/2);
    end
else
    B = A;
end

% Stretch out each edge so that it also exists in the coming edge_life time slices
B_orig = B;
for s = 2:edge_life
    %B(s:end, :, :) = (B(s:end, :, :)*(s-1) + B_orig(1:end-s+1, :, :))/s;
    B(s:end, :, :) = B(s:end, :, :) + B_orig(1:end-s+1, :, :);
end

% Add identity to all slices and normalize
u = repmat(1:N, 1, size(A,1)).';
v = repelem(1:size(A,1), 1, N).';
I = sptensor([v u u], ones(size(A,1)*N, 1));
C = B+I;
for t = 1:size(C,1)
    D = sparse(1:N, 1:N, 1./sqrt(sum(spmatrix(C(t, :, :)), 2)));
    C(t,:,:) = sptensor(D*spmatrix(C(t,:,:))*D);
    t
end

% Split up tensor
C_train = C(1:T, :, :);
C_val = C(1+no_val_samples : T+no_val_samples, :, :);
C_test = C(1+no_val_samples+no_test_samples : end, :, :);

% Compute M products
Ct_train = reshape(sptensor(sparse(M)*spmatrix(reshape(C_train, [T, N^2]))), [T, N, N]);
Ct_val = reshape(sptensor(sparse(M)*spmatrix(reshape(C_val, [T, N^2]))), [T, N, N]);
Ct_test = reshape(sptensor(sparse(M)*spmatrix(reshape(C_test, [T, N^2]))), [T, N, N]);

%% Save data to files

A_subs = A.subs;
A_vals = A.vals;
A_labels_subs = A_labels.subs;
A_labels_vals = A_labels.vals;
C_subs = C.subs;
C_vals = C.vals;
C_train_subs = C_train.subs;
C_train_vals = C_train.vals;
C_val_subs = C_val.subs;
C_val_vals = C_val.vals;
C_test_subs = C_test.subs;
C_test_vals = C_test.vals;
Ct_train_subs = Ct_train.subs;
Ct_train_vals = Ct_train.vals;
Ct_val_subs = Ct_val.subs;
Ct_val_vals = Ct_val.vals;
Ct_test_subs = Ct_test.subs;
Ct_test_vals = Ct_test.vals;

save([save_file_location, save_file_name], 'tensor_idx', 'tensor_labels', 'A_labels_subs', 'A_labels_vals', 'A_subs', 'A_vals', 'C_subs', 'C_vals', 'C_train_subs', 'C_train_vals', 'C_val_subs', 'C_val_vals', 'C_test_subs', 'C_test_vals', 'Ct_train_subs', 'Ct_train_vals', 'Ct_val_subs', 'Ct_val_vals', 'Ct_test_subs', 'Ct_test_vals', 'M')