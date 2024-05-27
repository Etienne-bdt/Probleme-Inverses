%% Déclaration des variables
clear all
close all


ProxL1 = @(x,gamma) sign(x).*max(0,abs(x)-gamma);
fe = 500;
N = 2^8;
J = 50;
w = 0.1;
fj = (1:J)/fe;
vect_t = (1:N)';
nfft = (-N/2:N/2-1)*fe/N;

Bw = randn(1,J)<w;
V = normrnd(0,1,1,J);

a = Bw.*V;

x = sum(a.*cos(2*pi*fj.*(1:N)'),2);


%% Affichage de x


%plot(1:N, x)

%%
Sx = abs(fft(x)).^2;



figure;
semilogy(Sx)
legend('Densité spectrale de x')
%% Creation de H

Iwav = 7;
Lwav = 9;

[h, ~] = morlet(-Iwav,Iwav,Lwav);
Hfull = convmtx(h',N);
H = Hfull(Lwav/2:end-Lwav/2,:);

vp_H = eig(H);
cond = max(abs(vp_H))/min(abs(vp_H))
%% Génération de y

ybar = H*x;

SNRdB = 20;
Px = 1/N*norm(x)^2;
sigma2 = 10^(-SNRdB/10)*Px;
b = sqrt(sigma2)*randn(N,1);
y_noisy = ybar+b;


%% Affichage de y
figure;
plot(y_noisy)
legend('Signal y bruité')
%% Least-Square Inversion

xLS = H\y_noisy;
figure;
plot(xLS);
legend('Inversion par moindres carrés')
%% Periodo de x Least Square

Sxls = abs(fft(xLS)).^2;

figure;
semilogy(nfft, Sx, nfft, Sxls);
legend('DSP de x', 'DSP de $x_{LS}$','interpreter','latex')
%Le problème est dur à résoudre, il y a un mauvais conditionnement, par
%conséquent, la solution des moindres carrés ne marche pas bien

%% Laplacian Based Smoothing Regularization



%% Creation de la matrice de reguation
RegMat = toeplitz([2 -1 zeros(1,N-2)]);
RegMat(1,:) = [];
RegMat(end,:) = [];
Pxreg = norm(RegMat,2);


%% Regularisation

Tlambda = 10.^(-5 : 0.2 : 2);
Nlambda = length(Tlambda);
Tab_MSE = zeros(1,Nlambda)+NaN;
Tab_J = zeros(1,Nlambda)+NaN;

figure;
i=0;
for lambda = Tlambda 
    i = i+1;
    x_hat = (H'*H + lambda * (RegMat'*RegMat))\H'*y_noisy;
    
    Tab_MSE(i) = norm(x-x_hat)^2;
    Tab_J(i) = norm(y_noisy-H*x_hat)^2+lambda * norm(RegMat*x_hat);
    
    % as a function of lambda
    subplot(311)
    plot(vect_t,x,'g')
    hold on
    plot(vect_t,x_hat,'k')
    plot(vect_t,y_noisy)
    axis tight
    xlabel('t')
    legend('x','xhat','y')
    title(['lambda = ' num2str(Tlambda(i))])    
    hold off
    subplot(312)
    loglog(Tlambda,Tab_MSE);
    xlabel('$\lambda$','interpreter','latex')
    ylabel('$||x-\hat{x}_{\lambda}||_2^2$','interpreter','latex')
    xlim([Tlambda(1) Tlambda(end)]);
    subplot(313)
    loglog(Tlambda,Tab_J);
    xlabel('$\lambda$','interpreter','latex')
    ylabel('$J(\hat{x}_{\lambda})$','interpreter','latex')
    xlim([Tlambda(1) Tlambda(end)]);
    drawnow
    
end

%% Lambda Optimal

[minerr, i_opt] = min(Tab_MSE);
lambda_opt = Tlambda(i_opt);
x_hat_opt = (H'*H + lambda_opt * (RegMat'*RegMat))\H'*y_noisy;

figure;
plot(vect_t, x_hat_opt)

%% DSP pour x_hat_opt

Sx_hat_opt = abs(fft(x_hat_opt)).^2;

%% Comparaison des dsp
figure;
semilogy(nfft, Sx, nfft,  Sx_hat_opt)
legend('Sx','Sx chapeau')

%% DCT based Regularization

psi = dctmtx(N);
dct = psi*x;
figure;
plot(dct)
legend('DCT de x')

%Beaucoup de coefficients sont proches de 0, on peut comprimer beaucoup une
%image/un signal en négligeant ces coefficients

%% Det of psi

det_psi = det(psi)

%Matrice inversible

%% Regularisation
% ||y- Hx||^2 + ||psix||^2
% HtH x + psiTpsi x

Tlambda_dct2 = 10.^(-5 : 0.2 : 1);
Nlambda_dct2 = length(Tlambda_dct2);
Tab_MSE_dct2 = zeros(1,Nlambda_dct2)+NaN;
figure;
i=0;
for lambda_dct2 = Tlambda_dct2
    i=i+1;
    x_hat_dct2 = (H'*H + lambda_dct2*eye(N))\H'*y_noisy;
    subplot(2,1,1)
    plot(vect_t,x);
    hold on
    plot(vect_t,x_hat_dct2,'k')
    axis tight
    xlabel('t')
    legend('x','$\hat{x}_{DCT l_2}$', 'interpreter', 'latex')
    hold off
    subplot(2,1,2)
    Tab_MSE_dct2(i) = norm(x-x_hat_dct2)^2;
    ylabel('$||x-\hat{x}_{\lambda}||_2^2$','interpreter','latex')
    plot(1:Nlambda_dct2,Tab_MSE_dct2)
    drawnow
end

[~ , i_opt_dct2] = min(Tab_MSE_dct2);
lambda_opt_dct2 = Tlambda_dct2(i_opt_dct2);

x_hat_dct2 = (H'*H + lambda_opt_dct2*eye(N))\H'*y_noisy;

%% L1

%opérateur proximall1

% Ici, ca revient à faire ce qu'on voit slide 160, on résoud le problème
% sur u (dct de x) pour ca on fait algo d'opti (voir BE opti pour ca)
Tlambda_dct = 10.^(-5 : 0.2 : 1);
Nlambda_dct = length(Tlambda_dct);
Tab_MSE_dct = zeros(1,Nlambda_dct)+NaN;
figure;
s=0.1;
i=0;
for lambda_dct = Tlambda_dct
    u = psi*y_noisy;
    i=i+1;
    for l=1:50
        u = ProxL1(u-s*(((H*psi')'*(H*psi'))*u-(H*psi')'*y_noisy),lambda_dct);
    end
    subplot(2,1,1)
    plot(vect_t,x);
    hold on
    plot(vect_t,psi'*u,'k')
    axis tight
    xlabel('t')
    legend('x','$\hat{x}_{DCT l_1}$','interpreter','latex')
    hold off
    subplot(2,1,2)
    Tab_MSE_dct(i) = norm(x-psi'*u)^2;
    ylabel('$||x-\Psi^Tu}||_2^2$','interpreter','latex')
    plot(1:Nlambda_dct,Tab_MSE_dct)
    drawnow
end

[~ , i_opt_dct] = min(Tab_MSE_dct);
lambda_opt_dct1 = Tlambda_dct(i_opt_dct);





%% Iteration sur lambda opt
k2max = 50;
k=1;
u = psi*y_noisy;
err_dct = zeros(1,k2max);
figure;
while k<=k2max
    u = ProxL1(u-s*(((H*psi')'*(H*psi'))*u-(H*psi')'*y_noisy),lambda_opt_dct1);
    subplot(2,1,1)
    plot(vect_t,x);
    hold on
    plot(vect_t,psi'*u,'k')
    axis tight
    xlabel('t')
    legend('x','$\Psi^Tu_{\lambda}$','interpreter','latex')
    hold off
    subplot(2,1,2)
    err_dct(k) = norm(x - psi'*u)^2;
    ylabel('$||x-\Psi^Tu_{\lambda}||_2^2$','interpreter','latex')
    plot(err_dct)
    k=k+1;
    drawnow
end

x_hat_dct1 = psi'*u;

%% Comparaison

figure;
hold on
plot(x)
plot(x_hat_dct2)
plot(x_hat_dct1)
hold off
legend('x', '$\hat{x}_{DCT2}$','$\hat{x}_{DCT1}$','interpreter','latex')
%% DSP

Sx_hat_dct2 = abs(fft(x_hat_dct2)).^2;
Sx_hat_dct1 = abs(fft(x_hat_dct1)).^2;

%% DSP comp

%figure;
%semilogy(nfft, Sx, nfft,  Sx_hat_dct2, nfft, Sx_hat_dct1)
%legend('Sx','Sx dct1' , 'Sx dct2')

%% Dictionary-based regularizations
K=2^9;

vk = linspace(1/fe,100/fe,K);
vkn = (1:N)'*vk;

D = cos(2*pi*vkn);

% ||y - HDz||^2 + lambda||z||^2
% (HD)'HDz - HD'y + lambda z

%% L2

z_hat = ((H*D)'*(H*D)+ eye(K))\(H*D)'*y_noisy;

x_hat_dict2 = D*z_hat;

figure;
hold on
plot(x)
plot(x_hat_dict2)
hold off

%Simple, rapide, efficace

%% L1 mais lambda


Tlambda_dict = 10.^(-5 : 0.2 : 1);
Nlambda_dict = length(Tlambda_dict);
Tab_MSE_dict = zeros(1,Nlambda_dict)+NaN;
i=0;
figure;
s=0.001;
for lambda_dict = Tlambda_dict
    i = i+1;
    z = ones(K,1);
    for l=1:40
        z = ProxL1(z-s*(((H*D)'*(H*D))*z-(H*D)'*y_noisy),lambda_dict);
    end
    subplot(211)
    plot(vect_t,x)
    hold on
    plot(vect_t,D*z,'k')
    axis tight
    xlabel('t')
    legend('x','$Dz l_1$','interpreter' ,'latex')
    hold off
    subplot(2,1,2)
    Tab_MSE_dict(i) = norm(x-D*z)^2;
    plot(Tab_MSE_dict)
    drawnow
end


[minerr_dict, i_opt_dict] = min(Tab_MSE_dict);
lambda_opt_dict1 = Tlambda_dict(i_opt_dict);



%% L1

k=1;
z = ones(K,1);
err_dic = zeros(1,k2max);
figure;
while k<=k2max
    z = ProxL1(z-s*(((H*D)'*(H*D))*z-(H*D)'*y_noisy),lambda_opt_dict1);
    subplot(2,1,1)
    plot(vect_t,x);
    hold on
    plot(vect_t,D*z,'k')
    axis tight
    xlabel('t')
    legend('x','$Dz_{\lambda l_1}$', 'interpreter','latex')
    hold off
    subplot(2,1,2)
    err_dic(k) = norm(x-D*z)^2;
    ylabel('$||x-Dz_{\lambda}||_2^2$','interpreter','latex')
    k=k+1;
    plot(1:k2max, err_dic)
    drawnow
end

x_hat_dict1 = D*z;