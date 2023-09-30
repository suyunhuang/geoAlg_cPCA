function v_c = geocpca(targetX, BG, NumPCs, alpha,...
    max_iter, tau, beta1, beta2, NN, u_ini)

%% NN: random subset size used for approximate gradient in armijofast.m
%%     default NN=2000
%% u_ini: initial PCs

p=size(targetX,2);
n=size(targetX,1);
m=size(BG,1);
method='cayley'; % currently we only have method 'cayley' 
                 % for retraction map from tangent space to manifold

if nargin <= 8
        u_c  = orth(normrnd(0,1,[p,NumPCs]));  % random initial 
        NNum=2000; 
end

obj_tc = zeros(max_iter,1);
stepsize_tc = zeros(max_iter,1);
innerIter_tc = zeros(max_iter,1);
for iter = 1:max_iter
    % show the process of computation
    if (iter == 1); tic;fprintf('Cayley transform:\n');end
    if mod(iter,5) == 0;fprintf([num2str(iter),'/',num2str(max_iter),'\n']);end
    % the gradient of the objective function
    grad = (targetX')*(targetX*u_c)/n...
        -alpha*(BG')*(BG*u_c)/m;
    % the value of the objective function at current point
    obj_tc(iter) = trace(u_c'*grad);
    % the projected gradient of the objective function
    % proj_grad = grad - u_c*(grad'*u_c);
    % determine the stepsize by using Armijo algorithm
    [u_c, tau, inner_iter] = armijofast(u_c, grad, tau, beta1, beta2, targetX, BG, alpha, method,NNum);
    stepsize_tc(iter) = tau;
    innerIter_tc(iter) = inner_iter;
end
v_c = u_c;
end


