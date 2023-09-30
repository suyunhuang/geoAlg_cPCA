%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% contrastive PCA: 
%% original algorithm by EVD vs. the proposed geometric algorithm
%% demo example using MNIST dataset with grass as backgroud
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%% loading and preparing for data
%
% * mnist.mat contains training set of 60,000 examples and test set of 10,000
%   examples, each of size 28 x 28.
% * BG.mat is the dataset of 30 grass images, each image has size 28 x 28.
% * size(trainX) = 60000 x 784; size(trainY) = 1 x 60000
% * size(testX)  = 10000 x 784; size(testY)  = 1 x 10000;
% * size(BG) = 28 x 28 x 30;
% * type of trainX, trainY, testX, testY: uint8
% * type of BG: double

load('BG.mat');
load('mnist.mat');

%% parameters setting
% * d = 28 : d x d is the size of each image
% * m = 30 : the sample size of background dataset
% * n = 60000 : the sample size of trainX
% * nt = 10000  : the sample size of testX

[d,~,m] = size(BG);
[n,~]   = size(trainX);
[nt,~]  = size(testX);
NumPCs=150; % number of components

rr= 1; % create large images by pasting rr x rr images
% set rr =1,2,3,
% e.g., rr =2, the enlarged image consists of 2x2 original images with
% background (2 x 2 backgrounds are randomly drawn from BG database and
% added to the original image)
% restriction: rr<=3 for the cPCA via eigenvalue decomposition on my MacBook

%% generating images of "digits on grass"
BG_tmp= zeros(rr*d,rr*d,m);
for i=1:m
    tmp = zeros(rr*d,rr*d);
    for j = 1:rr
        for k = 1:rr
            tmp(d*(j-1)+1:d*j,d*(k-1)+1:d*k) = BG(:,:,randi([1,m]));
        end
    end
    BG_tmp(:,:,i) = tmp;
end
BG_enlarged = reshape(BG_tmp,[d^2*rr^2,m])'; % prepare BG: reshaped and transposed
BG_cent = BG_enlarged - repmat(mean(BG_enlarged), [m,1]); % centered
clear BG_tmp

% trainX_tmp consists of enlarged images by pasting rr x rr original images
% reshape to n x (rr*d)^2

trainX_tmp= zeros(n,rr^2*d^2);

for i=1:n
    tmp=reshape(trainX(i,:),d,d);
    trainX_tmp(i,:)=reshape(repmat(tmp,rr),d^2*rr^2,1);
end

x_clear=trainX_tmp;

testX_tmp= zeros(size(testX,1),rr^2*size(testX,2));

for i=1:size(testX,1)
    tmp=reshape(testX(i,:),d,d);
    testX_tmp(i,:)=reshape(repmat(tmp,rr),d^2*rr^2,1)';
end

x_tst_clear=testX_tmp;
clear testX_tmp trainX_tmp


targetX = (x_clear + BG_enlarged(unidrnd(m,[n,1]),:))/2; %
testX = (x_tst_clear + BG_enlarged(unidrnd(m,[nt,1]),:))/2;
%  centralized data
mean_targetX=mean(targetX);
targetX_cent = targetX - repmat(mean_targetX,[n,1]);
x_tst_cent = testX - repmat(mean_targetX,[nt,1]);

%mean_tst_clear=mean(x_tst_clear);
%x_tst_clear_cent = x_tst_clear - repmat(mean_tst_clear,[nt,1]);

% Show the pictures of MNIST
ind = unidrnd(30,[nt,1]);


figure
sgtitle('Clear test images')
for i=1:30
    subplot(5,6,i)
    tmp = reshape(x_tst_clear(ind(i),:),[d*rr,d*rr]);
    imshow(uint8(tmp'))
end

figure
sgtitle('Test images with background grass')
for i=1:30
    subplot(5,6,i)
    tmp = reshape(testX(ind(i),:),[d*rr,d*rr]);
    imshow(uint8(tmp'))
end


%% cPCA method via eigenvalue decomposition 
alpha=1;
tic
% calculate the contrastive covariance matrix: M_target - alpha*M_BG
M_target = cov(targetX);
M_BG     = cov(BG_enlarged);
% extract the leading eigenvectors
[u_cpca_eig,~] = eigs(M_target-alpha*M_BG, NumPCs,"largestreal");
u_cpca_eig = u_cpca_eig(:,end-NumPCs+1:end);
original_alg_time = toc

x_tst_tmp= testX - repmat(mean_targetX,[nt,1]);
cpca_approx_eig = u_cpca_eig*(u_cpca_eig'*x_tst_tmp');

figure(3)
sgtitle('cPCA reconstruction via EVD')
for i=1:30
    subplot(5,6,i)
    tmp = reshape(cpca_approx_eig (:,ind(i))+mean_targetX',[d*rr,d*rr]);
    imshow(uint8(tmp'))
end


%% constrastive PCA by Cayley transform
% * optimization problem on Stiefel manifold
% * Stiefel manifold is defined by u'*u = eye(r)
% * the objective function is given by 0.5*trace(u'*M*u)
% * Conduct cPCA by cayley transform
% * Parameters
%    alpha : constrast parameter which represents the trade-off between the target
%            covariance and background covariance.
%    NumPCs : the abbreviation of number of components
%    tau_ini : the initial value of stepsize in the iteration process for the
%    max_iter : the maxima iteration number.
%    beta1 : 1_st parameter for Armijo algorithm
%    beta2 : 2_nd parameter for Armijo algorithm

max_iter =30;
beta1=0.005;beta2=0.9;
tic
[u_cpca, l_c] = geocpca(targetX_cent, BG_cent,NumPCs, alpha,...
    max_iter, 1, beta1, beta2);
time_our_alg= toc

x_tst_tmp= testX - repmat(mean_targetX,[nt,1]);
cpca_approx = u_cpca*(u_cpca'*x_tst_tmp');


figure(4)
sgtitle('cPCA reconstruction by our algorithm')
for i=1:30
    subplot(5,6,i)
    tmp = reshape(cpca_approx(:,ind(i))+mean_targetX',[d*rr,d*rr]);
    imshow(uint8(tmp'))
end

