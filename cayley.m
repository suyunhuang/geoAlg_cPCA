function U_n = cayley(GradF, U_c, tau)

% U_n = cayley(GradF,U_c,tau)
%     = (eye(p) + (tau/2)*W)*pinv(eye(p) - (tau/2)*W)*U_c
% where GradF is the projected gradient of F at U_c, tau is the stepsize

% Cayley transform:
% L = [GradF, U_c]; 
% R = [U_c,-GradF];
% W = GradF*U_c' - U_c*GradF' = L*R', 
% The Cayley transform is given by
% U_n = (eye(p) + (tau/2)*W   )*pinv(eye(p) - (tau/2)*W   )*U_c; 
%     = (eye(p) + (tau/2)*L*R')*pinv(eye(p) - (tau/2)*L*R')*U_c; 

% Sherman-Morrison-Woodbury identity:
% pinv(eye(p) - (tau/2)*L*R') =
% eye(p) + (tau/2)*L*pinv(eye(2*r) - (tau/2)*R'*L)*R'

[~,r] = size(U_c);
L = [GradF, U_c];
R = [U_c,-GradF];

U_n = U_c + tau*L*pinv(eye(2*r)-(tau/2)*(R'*L))*(R'*U_c);

end


