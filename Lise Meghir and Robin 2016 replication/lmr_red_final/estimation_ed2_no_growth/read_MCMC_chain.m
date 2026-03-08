clear;

alpha   = importdata('chain_param_10001_10000.dat');
s1   = importdata('chain_param_10002_10000.dat');
zeta = importdata('chain_param_10003_10000.dat');
f1   = importdata('chain_param_10004_10000.dat');
f2   = importdata('chain_param_10005_10000.dat');
f3   = importdata('chain_param_10006_10000.dat');
f4   = importdata('chain_param_10007_10000.dat');
f5   = importdata('chain_param_10008_10000.dat');
delta= importdata('chain_param_10009_10000.dat');
sigma= importdata('chain_param_10010_10000.dat');
beta = importdata('chain_param_10011_10000.dat');
b    = importdata('chain_param_10012_10000.dat');
c    = importdata('chain_param_10013_10000.dat');
f6    = importdata('chain_param_10014_10000.dat');
f7    = importdata('chain_param_10015_10000.dat');

objFun = importdata('chain_post_10000.dat');


for i=1:13
    
    
    alpha   = [ alpha    importdata(['chain_param_10001_',num2str(10000+i),'.dat'])];
    s1   = [ s1    importdata(['chain_param_10002_',num2str(10000+i),'.dat'])];
    zeta = [ zeta  importdata(['chain_param_10003_',num2str(10000+i),'.dat'])];
    f1   = [ f1    importdata(['chain_param_10004_',num2str(10000+i),'.dat'])];
    f2   = [ f2    importdata(['chain_param_10005_',num2str(10000+i),'.dat'])];
    f3   = [ f3    importdata(['chain_param_10006_',num2str(10000+i),'.dat'])];
    f4   = [ f4    importdata(['chain_param_10007_',num2str(10000+i),'.dat'])];
    f5   = [ f5    importdata(['chain_param_10008_',num2str(10000+i),'.dat'])];
    delta= [ delta importdata(['chain_param_10009_',num2str(10000+i),'.dat'])];
    sigma= [ sigma importdata(['chain_param_10010_',num2str(10000+i),'.dat'])];
    beta = [ beta  importdata(['chain_param_10011_',num2str(10000+i),'.dat'])];
    b = [ b  importdata(['chain_param_10012_',num2str(10000+i),'.dat'])];
    c = [ c  importdata(['chain_param_10013_',num2str(10000+i),'.dat'])];
    f6 = [ f6  importdata(['chain_param_10014_',num2str(10000+i),'.dat'])];
    f7 = [ f7  importdata(['chain_param_10015_',num2str(10000+i),'.dat'])];
    
    
  objFun = [objFun importdata(['chain_post_',num2str(10000+i),'.dat'])];
  
end

alpha = alpha';
s1 = s1';
zeta = zeta';
beta = beta';
f1 = f1';
f2 = f2';
f3 = f3';
f4 = f4';
f5 = f5';
f6 = f6';
f7 = f7';
sigma = sigma';
delta = delta';
b = b';
c = c';

objFun = objFun';

% transform values and use only the last 1000*95
% alpha_ = exp(alpha(end-500:end,:));
% s1_    = exp(s1(end-500:end,:));
% zeta_   =  0.1*exp(zeta(end-500:end,:))./(exp(zeta(end-500:end,:))+exp(-zeta(end-500:end,:)));
% beta_   = exp(beta(end-500:end,:))./(exp(beta(end-500:end,:))+exp(-beta(end-500:end,:)));
% f1_     = exp(f1(end-500:end,:));
% f2_     = exp(f2(end-500:end,:))./(exp(f2(end-500:end,:))+exp(-f2(end-500:end,:))); 
% f3_     = 1 - exp(f3(end-500:end,:));
% f4_     = exp(f4(end-500:end,:));
% f5_     = exp(f5(end-500:end,:));
% f6_     = exp(f6(end-500:end,:));
% f7_     = exp(f7(end-500:end,:));
% delta_  = 0.5*exp(delta(end-500:end,:))./(exp(delta(end-500:end,:))+exp(-delta(end-500:end,:))) ;
% sigma_  = exp(sigma(end-500:end,:));
% b_      = exp(b(end-500:end,:))./(exp(b(end-500:end,:))+exp(-b(end-500:end,:)));
% c_      = exp(c(end-500:end,:));

% transform values and use only the last 1000*95
alpha_ = exp(alpha(end-2000:end,:));
s1_    = exp(s1(end-2000:end,:));
zeta_   =  0.1*exp(zeta(end-2000:end,:))./(exp(zeta(end-2000:end,:))+exp(-zeta(end-2000:end,:)));
beta_   = exp(beta(end-2000:end,:))./(exp(beta(end-2000:end,:))+exp(-beta(end-2000:end,:)));
f1_     = exp(f1(end-2000:end,:));
f2_     = exp(f2(end-2000:end,:))./(exp(f2(end-2000:end,:))+exp(-f2(end-2000:end,:))); 
f3_     = 1 - exp(f3(end-2000:end,:));
f4_     = exp(f4(end-2000:end,:));
f5_     = exp(f5(end-2000:end,:));
f6_     = exp(f6(end-2000:end,:));
f7_     = exp(f7(end-2000:end,:));
delta_  = 0.5*exp(delta(end-2000:end,:))./(exp(delta(end-2000:end,:))+exp(-delta(end-2000:end,:))) ;
sigma_  = exp(sigma(end-2000:end,:));
b_      = exp(b(end-2000:end,:))./(exp(b(end-2000:end,:))+exp(-b(end-2000:end,:)));
c_      = exp(c(end-2000:end,:));

disp('eta');
[mean(alpha_(:)) std(alpha_(:))]
disp('s');
[mean(s1_(:)) std(s1_(:))]
disp('zeta');
[mean(zeta_(:)) std(zeta_(:))]
disp('delta');
[mean(delta_(:)) std(delta_(:))]
disp('b');
[mean(b_(:)) std(b_(:))]
disp('c');
[mean(c_(:)) std(c_(:))]
disp('A');
[mean(f1_(:)) std(f1_(:))]
disp('alpha');
[mean(f2_(:)) std(f2_(:))]
disp('rho');
[mean(f3_(:)) std(f3_(:))]
disp('a_x');
[mean(f4_(:)) std(f4_(:))]
disp('b_x');
[mean(f6_(:)) std(f6_(:))]
disp('a_y');
[mean(f5_(:)) std(f5_(:))]
disp('b_y');
[mean(f7_(:)) std(f7_(:))]
disp('beta');
[mean(beta_(:)) std(beta_(:))]
disp('sigma');
[mean((sigma_(:).^2)) std((sigma_(:).^2))]















