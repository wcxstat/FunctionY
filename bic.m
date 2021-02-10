function [res1,res2]=bic(Y,U,Zcov,lam,eig,xi,arg,rang1,rang2)
n=length(U);p=size(Zcov,1)/n;L=length(arg);
pi_h=mean(U); % estimation of pi
muY0=(1 - U)*Y/(n*(1-pi_h));
muY1=U*Y/(n*pi_h);
muZ0=zeros(p, L); muZ1=zeros(p, L); % p-by-L matrix
for i=1:p
    index=((i-1)*n+1):(i*n);
    muZ0(i,:)=(1-U)*Zcov(index,:)/(n*(1-pi_h));
    muZ1(i,:)=U*Zcov(index,:)/(n*pi_h);
end

bic_score=zeros(length(rang1),length(rang2));
for i1=1:length(rang1)
    for j1=1:length(rang2)
        m1=rang1(i1);m2=rang2(j1);
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
        % b_hat = eig(:,1:m1) * bcoef_hat * eig(:,1:m2)';
        % mesh(arg,arg,b_hat) % estimation
        % hold on
        % mesh(arg,arg,tb) % true
        % mise_vec(iter) = trapz2((b_hat-tb).^2,arg,arg); % integrate squared error(MSE)
        
        mu0 = muY0 - beta_hat' * muZ0;
        mu1 = muY1 - beta_hat' * muZ1;
        rX_hat = repmat(((1-pi_h)*mu0+pi_h*theta_hat*mu1)/(1-pi_h+pi_h*theta_hat^2),n,1)+xi(:,1:m1)*bcoef_hat*eig(:,1:m2)';
        utheta_hat = 1 - U + U * theta_hat;
        linpart_hat = zeros(n, L);
        for i = 1:n
            ind4 = i:n:((p-1)*n+i);
            linpart_hat(i,:) = beta_hat' * Zcov(ind4,:);
        end
        error_hat = Y - repmat(utheta_hat',1,L) .* rX_hat - linpart_hat;
        bic_score(i1,j1)=n*log(mean(trapz(arg,(error_hat.^2)')))+log(n)*(m1*m2);
    end
end
[aa,bb]=min(bic_score);
[~,cc]=min(aa);
res2=cc+min(rang2)-1;res1=bb(cc)+min(rang1)-1;