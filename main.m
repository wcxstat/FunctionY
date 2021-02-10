addpath(genpath('/Users/xuwenchao/Documents/PACE_matlab'))
n=1000; % sample size
alpha=1.2; gamma1=3.0; gamma2=3.0; % smoothness of functions
L=100; arg=linspace(0,1,L);

Phi=ones(50,L); % eigenfunctions for a functional predictor
for i=2:50
    Phi(i,:)=sqrt(2)*cos((i-1)*pi*arg);
end
eigval=(1:50).^(-alpha/2); % root-eigenvalues
% eigval.^2 % eigenvalues

M=2000;
param=zeros(3,M);param_new=zeros(3,M);
param_n=zeros(3,M);param_nn=zeros(3,M);
mise=zeros(2,M);
% chisqm=zeros(2,M);
for iter = 1:M
Um = unifrnd(-sqrt(3),sqrt(3),n,50); % n * 50 random matrix
score = repmat(eigval,n,1) .* Um; % PC scores
X = score * Phi; % functional data and n * L matrix
% plot(arg, X(1:10,:)')

% generating the random error process
Zerr = normrnd(0,1,n,50);
error = repmat((1:50).^(-1.1/2),n,1).* Zerr * Phi; % n * L random matrix
% plot(arg, error(1:10,:)')
% [lam_err, eig_err, xi_err] = FPCA_bal(error, arg, 15);

% true slope function
bcoef = zeros(50,50); % fourier coefficient
for j = 1:50
    for k = 1:50
        bcoef(j,k) = 4*(-1)^(j+k)*j^(-gamma1)*k^(-gamma2);
    end
end
bcoef(1,1)=0.3;
tb = Phi' * bcoef * Phi; % true slope function/50-by-50 matrix
% mesh(arg,arg,b_hat-tb)
% plot(arg,tb(1:50,:))

% generating a binomial random variable
pi_t=0.6;
U=binornd(1, pi_t, 1, n);
theta=1.5;

% generating the other covariates
utheta=1-U+U*theta;
zij1=1*(((1:50).^ (-gamma1)) * score') .* utheta; % 1 by n vector
zk1=(1:50).^ (-gamma2) * Phi; % 1 by 50 vector
V1=1*repmat((1:50).^(-1.1/2),n,1).* normrnd(0,1,n,50) * Phi;
Zcov1=zij1' * zk1 + V1;
%Zcov1=V1;

zij2 = 0.5 * (((1:50).^ (-gamma1)) * score') .* utheta; % 1 by n vector
zk2 = (1:50).^ (-gamma2) * Phi; % 1 by 50 vector
V2 = 0.5 * repmat((1:50).^(-1.1/2),n,1).* normrnd(0,1,n,50) * Phi;
Zcov2 = zij2' * zk2 + V2;
%Zcov2=V2;

Zcov = [Zcov1;Zcov2]; p = size(Zcov,1)/n;
% plot(arg, Zcov2(1:20,:)')

% true coefficient
beta = [3,-2];

% generating the functional responses
Y = repmat(utheta',1,L) .* (score * bcoef * Phi) + ...
    + Zcov1 * beta(1) + Zcov2 * beta(2) + error; % n * L matrix
% plot(arg, Y(1:10,:))


% Functional PCA
[lam, eig, xi]=FPCA_bal(X,arg,20);
% plot(arg,eig(:,1:5))


pi_h = mean(U); % estimation of pi
muY0 = (1 - U) * Y / (n * (1 - pi_h));
muY1 = U * Y / (n * pi_h);
muZ0 = zeros(p, L); muZ1 = zeros(p, L); % p-by-L matrix
for i = 1:p
    index = ((i-1)*n+1) : (i*n);
    muZ0(i,:) = (1 - U) * Zcov(index,:) / (n * (1 - pi_h));
    muZ1(i,:) = U * Zcov(index,:) / (n * pi_h);
end

% m1 = 20; m2 = 20;
[m1,m2]=bic(Y,U,Zcov,lam,eig,xi,arg,2:6,2:6);
etaY = zeros(n, m2);
etaZ = zeros(n*p, m2);
for k = 1:m2
    etaY(:,k) = trapz(arg, Y .* repmat(eig(:,k)',n,1),2);
    etaZ(:,k) = trapz(arg, Zcov .* repmat(eig(:,k)',n*p,1),2);
end

gY0 = xi(:,1:m1)' * (etaY .* repmat((1-U)',1,m2))/(n*(1-pi_h)); % m1-by-m2 matrix
gY1 = xi(:,1:m1)' * (etaY .* repmat(U',1,m2))/(n*pi_h);
gZ0 = zeros(m1*p,m2); % m1*p-by-m2 matrix
gZ1 = zeros(m1*p,m2); % m1*p-by-m2 matrix
for i = 1:p
    index  = ((i-1)*n+1) : (i*n);
    ind = ((i-1)*m1+1) : (i*m1);
    gZ0(ind,:) = xi(:,1:m1)' * (etaZ(index,:) .* repmat((1-U)',1,m2))/(n*(1-pi_h));
    gZ1(ind,:) = xi(:,1:m1)' * (etaZ(index,:) .* repmat(U',1,m2))/(n*pi_h);
end

E00 = trapz(arg,muY0.^2) + lam(1:m1).^(-1) * (gY0 .^2) * ones(m2,1);
E01 = trapz(arg,muY0 .* muY1) + lam(1:m1).^(-1) * (gY0 .* gY1) * ones(m2,1);
E10 = trapz(arg,muY1 .* muY0) + lam(1:m1).^(-1) * (gY1 .* gY0) * ones(m2,1);
E11 = trapz(arg,muY1.^2) + lam(1:m1).^(-1) * (gY1 .^2) * ones(m2,1);

F00 = zeros(p,1); F01 = zeros(p,1); F10 = zeros(p,1); F11 = zeros(p,1);
G00 = zeros(p,p); G01 = zeros(p,p); G10 = zeros(p,p); G11 = zeros(p,p);
Sigma_Z = zeros(p); % covariance of Z(t)
Sigma_YZ = zeros(p,1); % covariance of Y(t) and Z(t)
for i = 1:p
    index  = ((i-1)*n+1) : (i*n);
    ind = ((i-1)*m1+1) : (i*m1);
    F00(i) = trapz(arg, muY0 .* muZ0(i,:)) + lam(1:m1).^(-1) *(gY0 .* gZ0(ind,:)) *ones(m2,1);
    F01(i) = trapz(arg, muY0 .* muZ1(i,:)) + lam(1:m1).^(-1) *(gY0 .* gZ1(ind,:)) *ones(m2,1);
    F10(i) = trapz(arg, muY1 .* muZ0(i,:)) + lam(1:m1).^(-1) *(gY1 .* gZ0(ind,:)) *ones(m2,1);
    F11(i) = trapz(arg, muY1 .* muZ1(i,:)) + lam(1:m1).^(-1) *(gY1 .* gZ1(ind,:)) *ones(m2,1);
    Sigma_YZ(i) = trapz(arg, ones(1,n) * (Y .* Zcov(index,:))/n);
end
for i = 1:p
    for j = 1:p
        index1 = ((i-1)*m1+1) : (i*m1);
        index2 = ((j-1)*m1+1) : (j*m1);
        ind1 = ((i-1)*n+1) : (i*n);
        ind2 = ((j-1)*n+1) : (j*n);
        G00(i,j) = trapz(arg, muZ0(i,:) .* muZ0(j,:)) + lam(1:m1).^(-1) *(gZ0(index1,:) .* gZ0(index2,:)) *ones(m2,1);
        G01(i,j) = trapz(arg, muZ0(i,:) .* muZ1(j,:)) + lam(1:m1).^(-1) *(gZ0(index1,:) .* gZ1(index2,:)) *ones(m2,1);
        G10(i,j) = trapz(arg, muZ1(i,:) .* muZ0(j,:)) + lam(1:m1).^(-1) *(gZ1(index1,:) .* gZ0(index2,:)) *ones(m2,1);
        G11(i,j) = trapz(arg, muZ1(i,:) .* muZ1(j,:)) + lam(1:m1).^(-1) *(gZ1(index1,:) .* gZ1(index2,:)) *ones(m2,1);
        Sigma_Z(i,j) = trapz(arg, ones(1,n) * (Zcov(ind1,:) .* Zcov(ind2,:))/n);
    end
end

theta_h = zeros(1,11); beta_h = zeros(2,10);
% theta_h=repmat(1.5,1,11);beta_h=repmat([3;-2],1,10);
for k = 1:10
    Mat = Sigma_Z - ((1-pi_h)^2*G00+pi_h*(1-pi_h)*theta_h(k)*(G01+G10)+pi_h^2*theta_h(k)^2*G11)/(1-pi_h+pi_h*theta_h(k)^2);
    MatY = Sigma_YZ - ((1-pi_h)^2*F00+pi_h*(1-pi_h)*theta_h(k)*(F01+F10)+pi_h^2*theta_h(k)^2*F11)/(1-pi_h+pi_h*theta_h(k)^2);
    beta_h(:,k) = Mat\MatY;
    qu1 = (E01-(F01+F10)'*beta_h(:,k)+beta_h(:,k)'*G01*beta_h(:,k))*pi_h;
    qu2 = (E00-2*F00'*beta_h(:,k) +beta_h(:,k)'*G00*beta_h(:,k))*(1-pi_h)-pi_h*(E11-2*F11'*beta_h(:,k)+beta_h(:,k)'*G11*beta_h(:,k));
    qu3 = (E01-(F01+F10)'*beta_h(:,k)+beta_h(:,k)'*G01*beta_h(:,k))*(pi_h-1);
    theta_h(k+1) = (-qu2+sqrt(qu2^2-4*qu1*qu3))/(2*qu1);
end
theta_hat = theta_h(10);
beta_hat = beta_h(:,10);

bcoef_hat = zeros(m1,m2);
for j = 1:m1
    ind3 = j:m1:((p-1)*m1+j);
    g0 = gY0(j,:) - beta_hat' * gZ0(ind3,:);
    g1 = gY1(j,:) - beta_hat' * gZ1(ind3,:);
    bcoef_hat(j,:) = (lam(j))^(-1)*((1-pi_h)*g0 + theta_hat*pi_h*g1)/(1-pi_h+pi_h*theta_hat^2);
end
b_hat = eig(:,1:m1) * bcoef_hat * eig(:,1:m2)';
% mesh(arg,arg,b_hat) % estimation
% hold on
% mesh(arg,arg,tb) % true
% mise_vec(iter) = trapz2((b_hat-tb).^2,arg,arg); % integrate squared error(MSE)

mu0=muY0-beta_hat'*muZ0;
mu1=muY1-beta_hat'*muZ1;
rX_hat=repmat(((1-pi_h)*mu0+pi_h*theta_hat*mu1)/(1-pi_h+pi_h*theta_hat^2),n,1)+xi(:,1:m1)*bcoef_hat*eig(:,1:m2)';
utheta_hat=1-U+U*theta_hat;
linpart_hat=zeros(n, L);
for i = 1:n
    ind4 = i:n:((p-1)*n+i);
    linpart_hat(i,:) = beta_hat' * Zcov(ind4,:);
end
error_hat = Y - repmat(utheta_hat',1,L) .* rX_hat - linpart_hat;
% [lam1, eig1, xi1] = FPCA_bal(error_hat, arg, 15);

fZ0 = zeros(n*p,L);fZ1 = zeros(n*p,L);
for i = 1:p
    index1 = ((i-1)*m1+1) : (i*m1);
    ind1 = ((i-1)*n+1) : (i*n);
    fZ0(ind1,:) = repmat(muZ0(i,:),n,1) + (xi(:,1:m1) .* repmat(lam(1:m1).^(-1),n,1)) * gZ0(index1,:) * eig(:,1:m2)';
    fZ1(ind1,:) = repmat(muZ1(i,:),n,1) + (xi(:,1:m1) .* repmat(lam(1:m1).^(-1),n,1)) * gZ1(index1,:) * eig(:,1:m2)';
end
fZ = ((1-pi_h)*fZ0 + theta_hat*pi_h*fZ1)/(1-pi_h+pi_h*theta_hat^2);
V = Zcov - fZ .* repmat(repmat(utheta_hat',1,L),p,1);

ruo_hat = pi_h*(1-pi_h)/(1-pi_h+pi_h*theta_hat^2);
% ruo_t = pi_t*(1-pi_t)/(1-pi_t+pi_t*theta^2);
omega11 = ruo_hat^(-1)*mean((trapz(arg,(error_hat .* rX_hat)')).^2);
Gamma11 = theta_hat^2 * var(trapz(arg, (rX_hat.^2)'))/(pi_h*(1-pi_h));
omega21 = zeros(p,1);
Gamma21 = zeros(p,1);
for i = 1:p
    ind1 = ((i-1)*n+1) : (i*n);
    omega21(i) = mean(trapz(arg,(error_hat .* rX_hat)') .* trapz(arg,(error_hat .* V(ind1,:))') .* (U/pi_h-theta_hat*(1-U)/(1-pi_h)))...
        *ruo_hat^(-1);
    Gamma21(i) = ruo_hat^(-1)*theta_hat*(1-theta_hat^2)*(mean(trapz(arg, (rX_hat.^2)') .* trapz(arg, (rX_hat .* fZ(ind1,:))'))...
        -mean(trapz(arg, (rX_hat.^2)'))*mean(trapz(arg, (rX_hat .* fZ(ind1,:))')));
end
omega22 = zeros(p,p);
Gamma22 = zeros(p,p);
for i = 1:p
    for j = 1:p
        ind1 = ((i-1)*n+1) : (i*n);
        ind2 = ((j-1)*n+1) : (j*n);
        omega22(i,j)= ruo_hat^(-2)* mean(trapz(arg,(error_hat .* V(ind1,:))') .* trapz(arg,(error_hat .* V(ind2,:))'));
        Gamma22(i,j)= ruo_hat^(-2)*pi_h*(1-pi_h)*(1-theta_hat^2)^2*(mean(trapz(arg, (rX_hat .* fZ(ind1,:))') .* trapz(arg, (rX_hat .* fZ(ind2,:))'))...
            -mean(trapz(arg, (rX_hat .* fZ(ind1,:))'))*mean(trapz(arg, (rX_hat .* fZ(ind2,:))')));
    end
end

sigma11 = E00-2*beta_hat'* F00+beta_hat'*G00*beta_hat;
% sigma11 = ones(1,50)*(bcoef.^2 .* repmat((eigval.^2)',1,50))*ones(50,1);
sigma21 = F01-F10+(G01-G10)*beta_hat;
sigma22 = (Sigma_Z-((1-pi_h)^2*G00+pi_h*(1-pi_h)*theta_hat*(G01+G10)+pi_h^2*theta_hat^2*G11)/(1-pi_h+pi_h*theta_hat^2))/ruo_hat;

Sigma = [sigma11, sigma21';sigma21, sigma22];
Omega = [omega11, omega21';omega21, omega22];
Gamma = [Gamma11, Gamma21';Gamma21, Gamma22];
var_mat1 = Sigma^(-1)*(Omega+Gamma)*Sigma^(-1);

delta1_new = theta_hat/(pi_h*(1-pi_h))*mean((U-pi_h).*(trapz(arg, (rX_hat.^2)')));
delta2_new = zeros(p,1);
for i = 1:p
    ind1 = ((i-1)*n+1) : (i*n);
    Irf = trapz(arg, (rX_hat .* fZ(ind1,:))');
    delta2_new(i) = ruo_hat^(-1)*(1-theta_hat^2)*mean(Irf.* (U-pi_h));
end

%%% estimation %%%
thebeta = [theta_hat;beta_hat];
thebeta_new =thebeta - Sigma^(-1)*[delta1_new;delta2_new];

bcoef_hatn = zeros(m1,m2);
theta_hatn = thebeta_new(1);
beta_hatn = thebeta_new(2:3);
for j = 1:m1
    ind3 = j:m1:((p-1)*m1+j);
    g0 = gY0(j,:) - beta_hatn' * gZ0(ind3,:);
    g1 = gY1(j,:) - beta_hatn' * gZ1(ind3,:);
    bcoef_hatn(j,:) = (lam(j))^(-1)*((1-pi_h)*g0 + theta_hatn*pi_h*g1)/(1-pi_h+pi_h*theta_hatn^2);
end
b_hatn = eig(:,1:m1) * bcoef_hatn * eig(:,1:m2)';
mu0n = muY0 - beta_hatn' * muZ0;
mu1n = muY1 - beta_hatn' * muZ1;
rX_hatn = repmat(((1-pi_h)*mu0n+pi_h*theta_hatn*mu1n)/(1-pi_h+pi_h*theta_hatn^2),n,1)+xi(:,1:m1)*bcoef_hatn*eig(:,1:m2)';
utheta_hatn = 1 - U + U * theta_hatn;
linpart_hatn = zeros(n, L);
for i = 1:n
    ind4 = i:n:((p-1)*n+i);
    linpart_hatn(i,:) = beta_hatn' * Zcov(ind4,:);
end
error_hatn = Y - repmat(utheta_hatn',1,L) .* rX_hatn - linpart_hatn;
% [lam1, eig1, xi1] = FPCA_bal(error_hat, arg, 15);

fZn = ((1-pi_h)*fZ0 + theta_hatn*pi_h*fZ1)/(1-pi_h+pi_h*theta_hatn^2);
Vn = Zcov - fZn .* repmat(repmat(utheta_hatn',1,L),p,1);

ruo_hatn = pi_h*(1-pi_h)/(1-pi_h+pi_h*theta_hatn^2);
% ruo_t = pi_t*(1-pi_t)/(1-pi_t+pi_t*theta^2);
omega11n = ruo_hatn^(-1)*mean((trapz(arg,(error_hatn .* rX_hatn)')).^2);
omega21n = zeros(p,1);
for i = 1:p
    ind1 = ((i-1)*n+1) : (i*n);
    omega21n(i) = mean(trapz(arg,(error_hatn .* rX_hatn)') .* trapz(arg,(error_hatn .* Vn(ind1,:))') .* (U/pi_h-theta_hatn*(1-U)/(1-pi_h)))...
        *ruo_hatn^(-1);
end
omega22n = zeros(p,p);
for i = 1:p
    for j = 1:p
        ind1 = ((i-1)*n+1) : (i*n);
        ind2 = ((j-1)*n+1) : (j*n);
        omega22n(i,j)= ruo_hatn^(-2)* mean(trapz(arg,(error_hatn .* Vn(ind1,:))') .* trapz(arg,(error_hatn .* Vn(ind2,:))'));
    end
end

sigma11n = E00-2*beta_hatn'* F00+beta_hatn'*G00*beta_hatn;
% sigma11 = ones(1,50)*(bcoef.^2 .* repmat((eigval.^2)',1,50))*ones(50,1);
sigma21n = F01-F10+(G01-G10)*beta_hatn;
sigma22n = (Sigma_Z-((1-pi_h)^2*G00+pi_h*(1-pi_h)*theta_hatn*(G01+G10)+pi_h^2*theta_hatn^2*G11)/(1-pi_h+pi_h*theta_hatn^2))/ruo_hatn;

Sigman = [sigma11n, sigma21n';sigma21n, sigma22n];
Omegan = [omega11n, omega21n';omega21n, omega22n];
var_mat2 = Sigman^(-1)*Omegan*Sigman^(-1);
%var_mat2 = Sigman^(-1)*Omega*Sigman^(-1);

param(:,iter) = thebeta;
param_new(:,iter) = thebeta_new;
param_n(:,iter) = sqrt(n)*diag(1./sqrt(diag(var_mat1)))*(thebeta-[theta,beta]');
param_nn(:,iter) = sqrt(n)*diag(1./sqrt(diag(var_mat2)))*(thebeta_new-[theta,beta]');
mise(:,iter)=[trapz2((b_hat-tb).^2,arg,arg);...
    trapz2((b_hatn-tb).^2,arg,arg)];
% chisqm(:,iter)=[(thebeta_new-[theta,beta]')'*inv(var_mat2)*(thebeta_new-[theta,beta]');...
%     (thebeta-[theta,beta]')'*inv(var_mat1)*(thebeta-[theta,beta]')];
end

[mean(abs(param_n) <1.96,2)';...
    mean(abs(param_nn) <1.96,2)']

[mean(param')-[theta,beta];std(param')]

[mean(param_new')-[theta,beta];std(param_new')]

mean(mise')

std(mise')

[f,xi] = ksdensity(n*chisqm(1,:));
ft = pdf('Chisquare',0:0.01:10,3);
plot(xi,f)
hold on 
plot(0:0.01:10,ft)
axis([0,10,0,0.3])


[f,xi] = ksdensity(param_n(1,:));
ft = pdf('norm',xi,0,1);
plot(xi,f)
hold on
plot(xi,ft)