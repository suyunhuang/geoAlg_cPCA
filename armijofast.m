function [U_n, tau_new, inner_iter, ArmijoVal, ArmijoFig] = ...
    armijofast(U_c, GradF_c, tau, beta1, beta2, ...
    targetX, BG, alpha,method,NN)

% The default inner iteration of Armijo algorithm is 20
max_inner_iter = 20;
tau_inner = tau;



[m, ~] = size(BG);
[n, ~] = size(targetX);

if(m>NN)
    %mr=randsample(m,100);
    mr=1:NN;
    BG=BG(mr,:);
    m=NN;
end
if(n>NN)
     %nr=randsample(n,100);
    nr=1:NN; 
    targetX=targetX(nr,:);
    n=NN;
end




% The projected gradient of F_\alpha
PGradF = GradF_c - U_c*(GradF_c'*U_c);

% <PGradF,PGradF>_c = tr(PGradF'*PGradF) - (1/2)*tr(U_c'*PGradF)
GradF_norm = norm(PGradF,'fro') - 0.5*norm(U_c'*PGradF,'fro');

% The initial next point by Cayley transform with tau = tau_inner
if strcmp(method, 'cayley')
    U_n = cayley(GradF_c, U_c, tau_inner);
else
    U_n = cangeo(GradF_c, U_c, tau_inner);
end

% The initial gradient of next point
GradF_n = targetX'*(targetX*U_n)/n - alpha*BG'*(BG*U_n)/m;

% Calculate the objective function value of current point and next point
ObjVal_c = 0.5*trace(U_c'*GradF_c);
ObjVal_n = 0.5*trace(U_n'*GradF_n);

% The initial value of inner iteration in Armijo algorithm
inner_iter = 1;


% The initial logical value that determines the initial stepsize is going
% to lengthen or shorten
% if ArmijoVal >= 0, the stepsize is going to lengthen. 
% Otherwise, stepsize is going to shorten
ArmijoVal = ObjVal_n - ObjVal_c - beta1*tau_inner*GradF_norm;
SuffIncr  = (ArmijoVal >= 0);


ObjVals   = ObjVal_n;
LineVals  = ObjVal_c + beta1*tau_inner*GradF_norm;
StepSizes = tau;

if SuffIncr == 1
    while SuffIncr == 1 && inner_iter <= max_inner_iter % the maximal number of sub-iteration
        % Stretch Stepsize
        tau_inner = tau_inner/beta2; 
        % Update the next point U_n
        if strcmp(method, 'cayley')
            U_n = cayley(GradF_c, U_c, tau_inner);
        else
            U_n = cangeo(GradF_c, U_c, tau_inner);
        end
        % Calculate the gradient of F_\alpha at U_n
        GradF_n = targetX'*(targetX*U_n)/n -alpha*BG'*(BG*U_n)/m;
        % Derive the value of objective function at U_n 
        ObjVal_n = 0.5*trace(U_n'*GradF_n);
        % Update SuffIncr
        SuffIncr = (ObjVal_n - ObjVal_c - beta1*tau_inner*GradF_norm >= 0);
        % Update for the sub-iteration number
        inner_iter = inner_iter + 1; 

        ObjVals   = [ObjVals, ObjVal_n];
        LineVals  = [LineVals, ObjVal_c + beta1*tau_inner*GradF_norm];
        StepSizes = [StepSizes, tau_inner];
    end
    % Update U_c
    tau_new = tau_inner*beta2;
else
    while SuffIncr == 0 && inner_iter <= max_inner_iter % the maximal number of sub-iteration 
        if inner_iter == 70; beta2 = beta2*0.001; end
        % Shorten Stepsize
        tau_inner  = tau_inner*beta2; 
        % Update the next point U_n
        if strcmp(method, 'cayley')
            U_n = cayley(GradF_c, U_c, tau_inner);
        else
            U_n = cangeo(GradF_c, U_c, tau_inner);
        end
        % Calculate the gradient of F_\alpha at U_n
        GradF_n = targetX'*(targetX*U_n)/n -alpha*BG'*(BG*U_n)/m;
        % Derive the value of objective function at U_n 
        ObjVal_n = 0.5*trace(U_n'*GradF_n);
        % Update SuffIncr
        SuffIncr  = (ObjVal_n - ObjVal_c - beta1*tau_inner*GradF_norm >= 0);
        % Update for the sub-iteration number
        inner_iter = inner_iter + 1;

        ObjVals = [ObjVal_n, ObjVals];
        LineVals = [ObjVal_c + beta1*tau_inner*GradF_norm, LineVals];
        StepSizes = [tau_inner, StepSizes];
    end
    % Update u_c
    tau_new = tau_inner;  
end

if strcmp(method, 'cayley')
    U_n = cayley(GradF_c, U_c, tau_new);
else
    U_n = cangeo(GradF_c, U_c, tau_new);
end

% Setup the figure
StepSizes = [0, StepSizes];
LineVals  = [ObjVal_c, LineVals];
ObjVals   = [ObjVal_c, ObjVals];

%{
ArmijoFig = figure('Visible','off');
hax = axes; 
% title([method+' transform',...
%     ' stepsize = ',num2str(tau_new),...
%     ' inner iter = ',num2str(inner_iter)])
hold on
plot(StepSizes, ObjVals,  'b-*')
plot(StepSizes, LineVals, 'r-*')

line([tau_new, tau_new],get(hax,'YLim'),'LineStyle','--','Color','g')
%}
end


