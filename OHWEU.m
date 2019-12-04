% test for nus
clc;
clear all;


%% load data


addpath('nus');

fid = fopen('nus_vector','rb');
N = 269648;
D = 4096;
X = fread(fid, [N,D], 'double');
fclose(fid);


tag = load('tag_v1.txt');
rmpath('nus');


%% data filter

% taking the most frequent 21 classes
s = sum(tag, 1);
[~, ind] = sort(-s);
tag = tag(:, ind(1:21));
s = sum(tag, 2);
ind = find(s ~= 0);
X = X(ind, :);
tag = tag(ind, :);

% pre-process by following MIH
% zero-mean
        mm = mean(X);
        X = bsxfun(@minus, X, mm);

% normalize
        s = ceil(size(X, 1) / 10);
        
        for i = 1 : 10
            st = (i - 1) * s + 1;
            ed = min(i * s, size(X,1));
            X(st:ed, :) = normr(X(st:ed, :));
        end
        

l = randperm(size(X, 1));
numQ = 2000;
Xtraining = X(l(1:size(X, 1) - numQ), :);
Xtest = X(l(size(X, 1) - numQ + 1 : end), :);
trainLabels = tag(l(1:size(X, 1) - numQ), :);
testLabels = tag(l(size(X, 1) - numQ + 1 : end), :);


printHot = zeros(100, 1);
% printHot(1:5) = 1;
% printHot(10:10:100) = 1;
printHot(100) = 1;
clear X;

%% single test
bits = [32];
mAPs = [];
training_time = [];
time_function = [];
Cs = [0.1];

methods = {'sym','asym'}; % sym : symmetric projection; asym : asymmetric projection
for methodi = 1 : length(methods)
    for ic = 1 : length(Cs)
        for ii = 1 : length(bits)
    
            bit = bits(ii);
            method   = methods{methodi};
            disp(method);
            switch (method)

    
        
        
            case 'sym' % query and data point using same projection
                %parameter
                class_size = 21;
                C = Cs(ic);
                para.nbits = bit;

                % LSH init
                d = bit;
                W = randn(d, para.nbits);
                W = W ./ repmat(diag(sqrt(W'*W))',d,1);
                para.W = W;

                d = size(Xtraining, 2);
                W = randn(d, para.nbits);
                W = W ./ repmat(diag(sqrt(W'*W))',d,1);
                para.QW = W;

                d = class_size;
                labelW = randn(d, para.nbits);
                labelW = labelW ./ repmat(diag(sqrt(labelW'*labelW))',d,1);

                [W, pc] = eigs(cov(double(Xtraining((1:300),:))), bit);
                addpath('itq');
                [Y, R] = ITQ(double(Xtraining((1:300),:))*W,50);
                rmpath('itq');

                chunkN = ceil(size(Xtraining, 1) / 100);
                for i = 1 : 100

                    st = (i - 1) * chunkN + 1;
                    ed = i * chunkN;
                    ed = min(ed, size(Xtraining, 1));


                    if i == 1



                        % initial result

                        BTSPLH_Y_train = sign(Xtraining * W * R) * para.W;
                        BTSPLH_Y_test = sign(Xtest  * W * R) * para.W;
                        BTSPLH_Y_train(BTSPLH_Y_train >= 0) = 1;
                        BTSPLH_Y_train(BTSPLH_Y_train < 0) = -1;

                        BTSPLH_Y_test(BTSPLH_Y_test >= 0) = 1;
                        BTSPLH_Y_test(BTSPLH_Y_test < 0) = -1;

                        [mAP] = cal_precision_multi_label_batch(BTSPLH_Y_train, BTSPLH_Y_test, trainLabels, testLabels);

                        mAPs = [mAPs, mAP]



                    end

                    for j = st : ed
                        ll = zeros(1,class_size);
                        ll(trainLabels(j,:)>0) = 1;
                        target_codes = sign(ll * labelW);
                        
                        % learning two projections
                        X = Xtraining(j,:);
                        t = ones(1, para.nbits) -X * para.QW .* target_codes ./ (X*X');
                        l = find(t > C);
                        t(l) = C;
                        para.QW = para.QW + bsxfun(@times, (X' * target_codes), t);                            
                        
                        X = sign(X * W * R);
                        t = ones(1, para.nbits) -X * para.W .* target_codes ./ (X*X');
                        l = find(t > C);
                        t(l) = C;
                        para.W = para.W + bsxfun(@times, (X' * target_codes), t);                            

                    end                            

                    if printHot(i) == 1
                        BTSPLH_Y_train = sign(Xtraining * W * R) * para.W;
                        BTSPLH_Y_test = sign(Xtest * W * R) * para.W;
                        BTSPLH_Y_train(BTSPLH_Y_train >= 0) = 1;
                        BTSPLH_Y_train(BTSPLH_Y_train < 0) = -1;

                        BTSPLH_Y_test(BTSPLH_Y_test >= 0) = 1;
                        BTSPLH_Y_test(BTSPLH_Y_test < 0) = -1;
                        [mAP] = cal_precision_multi_label_batch(BTSPLH_Y_train, BTSPLH_Y_test, trainLabels, testLabels);

                        mAPs = [mAPs, mAP]
                    end
                 end        

            case 'asym'

                class_size = 21;



                %parameter
                        
                C = Cs(ic);

                para.nbits = bit;

                % LSH init
                d = bit;
                W = randn(d, para.nbits);
                W = W ./ repmat(diag(sqrt(W'*W))',d,1);
                para.W = W;

                d = size(Xtraining, 2);
                W = randn(d, para.nbits);
                W = W ./ repmat(diag(sqrt(W'*W))',d,1);
                para.QW = W;


                d = class_size;
                labelW = randn(d, para.nbits);
                labelW = labelW ./ repmat(diag(sqrt(labelW'*labelW))',d,1);

                [W, pc] = eigs(cov(double(Xtraining((1:300),:))), bit);
                addpath('itq');
                [Y, R] = ITQ(double(Xtraining((1:300),:))*W,50);
                rmpath('itq');

                chunkN = ceil(size(Xtraining, 1) / 100);
                for i = 1 : 100

                        st = (i - 1) * chunkN + 1;
                        ed = i * chunkN;
                        ed = min(ed, size(Xtraining, 1));


                        if i == 1
                            % initial result

                            BTSPLH_Y_train = sign(Xtraining * W * R) * para.W;
                            BTSPLH_Y_test = Xtest * para.QW;
                            BTSPLH_Y_train(BTSPLH_Y_train >= 0) = 1;
                            BTSPLH_Y_train(BTSPLH_Y_train < 0) = -1;

                            BTSPLH_Y_test(BTSPLH_Y_test >= 0) = 1;
                            BTSPLH_Y_test(BTSPLH_Y_test < 0) = -1;


                            [mAP] = cal_precision_multi_label_batch(BTSPLH_Y_train, BTSPLH_Y_test, trainLabels, testLabels);

                            mAPs = [mAPs, mAP]

                        end
                        tic;
                        for j = st : ed
                            ll = zeros(1,class_size);
                            ll(trainLabels(j,:)>0) = 1;
                            target_codes = sign(ll * labelW);
                            
                            % learning two projections

                            X = Xtraining(j,:);
                            t = ones(1, para.nbits) -X * para.QW .* target_codes ./ (X*X');
                            l = find(t > C);
                            t(l) = C;
                            para.QW = para.QW + bsxfun(@times, (X' * target_codes), t);                            
                            
                            X = sign(X * W * R);
                            t = ones(1, para.nbits) -X * para.W .* target_codes ./ (X*X');
                            l = find(t > C);
                            t(l) = C;
                            para.W = para.W + bsxfun(@times, (X' * target_codes), t);                            

                        end                            
                        Tdata = toc;
                        time_function = [time_function, Tdata];
                        if printHot(i) == 1
                            BTSPLH_Y_train = sign(Xtraining * W * R) * para.W;
                            BTSPLH_Y_test = Xtest * para.QW;
                            BTSPLH_Y_train(BTSPLH_Y_train >= 0) = 1;
                            BTSPLH_Y_train(BTSPLH_Y_train < 0) = -1;

                            BTSPLH_Y_test(BTSPLH_Y_test >= 0) = 1;
                            BTSPLH_Y_test(BTSPLH_Y_test < 0) = -1;
                            [mAP] = cal_precision_multi_label_batch(BTSPLH_Y_train, BTSPLH_Y_test, trainLabels, testLabels);

                            mAPs = [mAPs, mAP]

                        end
                end                   
                    
                    
            end

        end  
    end
end