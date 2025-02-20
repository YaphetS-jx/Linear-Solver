function [x] = PL2R(A,b,x_guess,tol,max_iter,omega,m,p,PC)
% Periodic L2-Richardson (PL2R) MATLAB code
% Copyright (C) 2018 Material Physics & Mechanics Group at Georgia Tech.
% Authors: Phanish Suryanarayana, Xin Jing
% (In collaboration with John E. Pask.)
% Last Modified: 7 Feb 2020  
% Solves the system Ax = b
% Inputs: A       : square matrix (size N x N),
%         b       : right hand side column vector (size N x 1),
%         x_guess : initial guess (of size b),
%         tol     : convergence tolerance
%         max_iter: maximum number of iterations
%         omega   : relaxation parameter (for Richardson update)
%         m       : Anderson history, no. of previous iterations to be considered in extrapolation
%         p       : Perform Anderson extrapolation at every p th iteration
%         PC      : Preconditioner, PC=0 (none), 1 (Jacobi) or 2 (ILU)
% Output: x       : solution vector
% NOTE  : A and b are to be provided. Other input parameters can be passed as "[]", to use defaults.
% e.g.  : Run code as, tic;x=PL2R(A,b,[],[],[],[],[],[],[]);toc;
 
N=length(b);
%%%%%%%%%%%% Default Input Parameters %%%%%%%%%%%%
if isempty(x_guess)
    x_guess = ones(N,1);  % initial guess vector
end
if isempty(tol)
    tol = 1e-6;               % convergence tolerance
end
if isempty(max_iter)
    max_iter = 1e5;           % maximum number of iterations
end
if isempty(omega)
    omega = 1.0;               % relaxation parameter
end
if isempty(m)
    m = 13;                   % Anderson history, no. of previous iterations to be considered in extrapolation
end
if isempty(p)
    p = 11;                    % Perform Anderson extrapolation at every p th iteration
end
if isempty(PC)
    PC = 0;                   % No preconditioning
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PC==1
    D=diag(A); % Jacobi PC (AAJ)
    if nnz(D==0)~=0
        error('Error: Diagonal has zeros. Use a different PC.')
    end
elseif PC==2
    [L,U] = ilu(A,struct('type','nofill')); % ILU PC
end

nb = norm(b);
if nb==0
    nb=1;
    disp('Message: The RHS is zero. Calculating residual instead of relative residual.')
end

Xk=zeros(N,m);
AXk=zeros(N,m);

x_prev = x_guess;
count = 1;
Ax = A*x_prev;
res = b - Ax;
relres = norm(res)/nb;  % relative residual

while count<=max_iter && relres > tol
    
    % APPLY PC
    if PC==1
        res = res./D; % Jacobi PC (AAJ)
    elseif PC==2
        res = U\(L\(res)); % ILU PC
    end
    
    % STORE HISTORY
    dx = omega * res;
    x_new = x_prev + dx;
    
    Ax_prev = Ax;
    Ax = A * x_new;
    k = mod(count-2,m)+1;
    Xk(:,k) = dx;
    AXk(:,k) = Ax - Ax_prev;
    
     % UPDATE ITERATE
    if rem(count,p)==0 
        alpha = pinv(AXk'*AXk)*(AXk'*(b-Ax));
        
        x_new = x_new + Xk * alpha;
        Xk(:,k) = x_new - x_prev;
        Ax = Ax + AXk * alpha;
        AXk(:,k) = Ax - Ax_prev;
    end
    res = b - Ax;
    relres = norm(res)/nb;
    x_prev = x_new;
    count = count + 1 ;
end

x = x_prev;

if count-1 == max_iter
    fprintf('PL2R exceeded maximum iterations and converged to a relative residual of %g. \n',relres);
else
    fprintf('PL2R converged to a relative residual of %g in %d iterations.\n',relres,count-1);
end

end
