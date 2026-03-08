% plot_model and obtain starting values for planners' matching function

% import model objects after running "MatchingSortingWages_serial"
% with current best parameter estimates


f_xy = importdata('f_xy.dat');
S_xy = importdata('S_xy.dat');
h_xy = importdata('h_xy.dat');

% S_xy_planner = importdata('../planner_ed2/S_xy.dat');
 
x=linspace(0,1,size(f_xy,1))';
y=linspace(0,1,size(f_xy,1))';
yval_y = linspace(0.001, 0.999, size(f_xy,1))';
xval_x = linspace(0.001, 0.999, size(f_xy,1))';

X.m = size(f_xy,1);
X.n = size(f_xy,2);


figure(2000)
S1_xy = max(0,S_xy);
colormap('hsv');
lines = linspace(0, max(S1_xy(:)), 25);
h1 = contour(100*y,100*x, S1_xy, lines,  'LineWidth', 2 );set(gca,'FontSize', 16, 'FontName', 'times')
axis square
xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
saveas(gcf, 'ed2_S_xy.eps', 'epsc2');


% figure(11)
% mesh(x,y,(f_xy))
% figure(12)
% mesh(x,y,h_xy)
% figure(13)
% mesh(x,y,(S_xy))

figure(214)
h1 = contourf(100*y,100*x, S_xy, [0 1000000000000], 'LineWidth', 2 );
set(gca,'FontSize', 16, 'FontName', 'times')
axis square
%daspect([2 3 1]);
xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
% hold on 
% contourf(100*y,100*x, S_xy_planner, [0 1000000000000], 'LineWidth', 2,'LineStyle','--' );
% hold off
saveas(gcf, 'ed2_M_xy.eps', 'epsc2');


STOP

H = sum(h_xy(:));

figure(100)
mesh(100*y,100*x, (h_xy/H));

figure(15)
%h2 = contourf(100*y,100*x, log(h_xy/H), log([0.0001:(max(h_xy(:)/H)/25):0.00062568]) );
h2 = contourf(100*y,100*x, log(h_xy), 'LineWidth', 2) ;

set(gca,'FontSize', 16, 'FontName', 'times')
axis square
!daspect([2 3 1]);
xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
saveas(gcf, 'ed2_h_xy.eps', 'epsc2');
% 
% figure(15)
% h2 = contourf(100*y(51:end),100*x, log(h_xy(:,51:end)), log([0.0001:(max(h_xy(:))/25):max(h_xy(:))]) );
% set(gca,'FontSize', 16, 'FontName', 'times')
% axis square
% !daspect([2 3 1]);
% xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
% ylabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
% saveas(gcf, 'ed2_h_xy.eps', 'epsc2');
% 


%% look at quantiles

h_x = sum(h_xy, 2);
h_y = sum(h_xy);
h_x = h_x ./ sum(h_x);
h_y = h_y ./ sum (h_y);

h_xy = h_xy ./ sum(h_xy(:));

h_xcy = h_xy ./ repmat(h_y, 100, 1);

H_xcy = cumsum(h_xcy);

h_ycx = h_xy ./ repmat( h_x, 1, 100 );
H_ycx = cumsum(h_ycx, 2);

H_xcy(isnan(H_xcy)) = 1;

xcy_q10 = zeros(100,1);
xcy_q20 = zeros(100,1);
xcy_q30 = zeros(100,1);
xcy_q40 = zeros(100,1);
xcy_q50 = zeros(100,1);
xcy_q60 = zeros(100,1);
xcy_q70 = zeros(100,1);
xcy_q80 = zeros(100,1);
xcy_q90 = zeros(100,1);

ycx_q10 = zeros(100,1);
ycx_q20 = zeros(100,1);
ycx_q30 = zeros(100,1);
ycx_q40 = zeros(100,1);
ycx_q50 = zeros(100,1);
ycx_q60 = zeros(100,1);
ycx_q70 = zeros(100,1);
ycx_q80 = zeros(100,1);
ycx_q90 = zeros(100,1);


for i = 1:100
xcy_q10(i) = find( H_xcy(:,i) >= 0.10, 1, 'first' );
xcy_q20(i) = find( H_xcy(:,i) >= 0.20, 1, 'first' );
xcy_q30(i) = find( H_xcy(:,i) >= 0.30, 1, 'first' );
xcy_q40(i) = find( H_xcy(:,i) >= 0.40, 1, 'first' );
xcy_q50(i) = find( H_xcy(:,i) >= 0.50, 1, 'first' );
xcy_q60(i) = find( H_xcy(:,i) >= 0.60, 1, 'first' );
xcy_q70(i) = find( H_xcy(:,i) >= 0.70, 1, 'first' );
xcy_q80(i) = find( H_xcy(:,i) >= 0.80, 1, 'first' );
xcy_q90(i) = find( H_xcy(:,i) >= 0.90, 1, 'first' );
end


index_2 = zeros(100,1);
for i = 1:100
ycx_q10(i) = find( H_ycx(i,:) >= 0.10, 1, 'first' );
ycx_q20(i) = find( H_ycx(i,:) >= 0.20, 1, 'first' );
ycx_q30(i) = find( H_ycx(i,:) >= 0.30, 1, 'first' );
ycx_q40(i) = find( H_ycx(i,:) >= 0.40, 1, 'first' );
ycx_q50(i) = find( H_ycx(i,:) >= 0.50, 1, 'first' );
ycx_q60(i) = find( H_ycx(i,:) >= 0.60, 1, 'first' );
ycx_q70(i) = find( H_ycx(i,:) >= 0.70, 1, 'first' );
ycx_q80(i) = find( H_ycx(i,:) >= 0.80, 1, 'first' );
ycx_q90(i) = find( H_ycx(i,:) >= 0.90, 1, 'first' );
end

% figure(101)
% plot( 100*x, xcy_q10 , 100*x, xcy_q20 ,  100*x, xcy_q30 ,  100*x, xcy_q40 ,  100*x, xcy_q50 ,  ...
%     100*x, xcy_q60 ,  100*x, xcy_q70 ,  100*x, xcy_q80 ,  100*x, xcy_q90 );
% ylim([0, 100]);
% xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
% ylabel('Conditional Decile of Worker Type','FontSize',16, 'FontName', 'times');
% saveas(gcf, 'ed2_q_xcy.eps', 'epsc2');
% 
% figure(201)
% plot( 100*y, ycx_q10 ,   100*y, ycx_q20 ,   100*y, ycx_q30 ,   100*y, ycx_q40 ,   100*y, ycx_q50 ,  ...
%      100*y, ycx_q60 ,   100*y, ycx_q70 ,   100*y, ycx_q80 ,   100*y, ycx_q90 );
% ylim([0, 100]);
% xlabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
% ylabel('Conditional Decile of Firm Type','FontSize',16, 'FontName', 'times');
% saveas(gcf, 'ed2_q_ycx.eps', 'epsc2');
% 

%% Plot conditional CDF H(y|x=Xq) and H(x|y=Yq)

% centiles
% figure(9)
% set(gca,'FontSize', 16, 'FontName', 'times')
% daspect([2 3 1]);
% plot1 = plot(100*y, H_ycx(1,:), 100*y, H_ycx(round(0.1*X.m),:), 100*y, H_ycx(round(0.2*X.m),:), 100*y, H_ycx(round(0.3*X.m),:),  100*y, H_ycx(round(0.4*X.m),:), 100*y, H_ycx(round(0.5*X.m),:),  100*y, H_ycx(round(0.6*X.m),:), 100*y, H_ycx(round(0.7*X.m),:), 100*y, H_ycx(round(0.8*X.m),:),  100*y, H_ycx(round(0.9*X.m),:),  100*y, H_ycx(X.m,:),  'LineWidth',3 );set(gca,'FontSize', 16, 'FontName', 'times')
%  set(plot1(1), 'LineStyle','-');
%  set(plot1(2), 'LineStyle','--');
%  set(plot1(3), 'LineStyle','.');
%  set(plot1(4), 'LineStyle','-');
%  set(plot1(5), 'LineStyle','--');
%  set(plot1(6), 'LineStyle','.');
%  set(plot1(7), 'LineStyle','-');
%  set(plot1(8), 'LineStyle','--');
%  set(plot1(9), 'LineStyle','.');
%  set(plot1(10), 'LineStyle','-');
%  set(plot1(11), 'LineStyle','--');
% ylim([0, 1]);
% xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
% ylabel('CDF of Firm Type, conditional on decile of Worker Type','FontSize',16, 'FontName', 'times');
% %daspect([2 300 1]);
% saveas(gcf, 'ed2_H_ycx.eps', 'epsc2');
% 
% figure(10)
% set(gca,'FontSize', 16, 'FontName', 'times')
% daspect([2 3 1]);
% plot1 = plot( 100*x, H_xcy(:,1), 100*x, H_xcy(:,round(.1*X.n)), 100*x, H_xcy(:,round(.2*X.n)), 100*x, H_xcy(:,round(.3*X.n)), 100*x, H_xcy(:,round(.4*X.n)), 100*x, H_xcy(:,round(.5*X.n)), 100*x, H_xcy(:,round(0.6*X.n)), 100*x, H_xcy(:,round(0.7*X.n)) ,100*x, H_xcy(:,round(0.8*X.n)), 100*x, H_xcy(:,round(0.9*X.n)), 100*x, H_xcy(:,X.n),   'LineWidth',3 );set(gca,'FontSize', 16, 'FontName', 'times')
%  set(plot1(1), 'LineStyle','-');
%  set(plot1(2), 'LineStyle','--');
%  set(plot1(3), 'LineStyle','.');
%  set(plot1(4), 'LineStyle','-');
%  set(plot1(5), 'LineStyle','--');
%  set(plot1(6), 'LineStyle','.');
%  set(plot1(7), 'LineStyle','-');
%  set(plot1(8), 'LineStyle','--');
%  set(plot1(9), 'LineStyle','.');
%  set(plot1(10), 'LineStyle','-');
%  set(plot1(11), 'LineStyle','--');
% ylim([0, 1]);
% xlabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
% ylabel('CDF of Worker Type, conditional on decile of Firm Type','FontSize',16, 'FontName', 'times');
% %daspect([2 300 1]);
% saveas(gcf, 'ed2_H_xcy.eps', 'epsc2');

% quintiles
figure(9)
set(gca,'FontSize', 16, 'FontName', 'times')
daspect([2 3 1]);
plot1 = plot(100*y, H_ycx(1,:), 100*y, H_ycx(round(0.2*X.m),:),  100*y, H_ycx(round(0.4*X.m),:),  100*y, H_ycx(round(0.6*X.m),:),  100*y, H_ycx(round(0.8*X.m),:),    100*y, H_ycx(X.m,:),  'LineWidth',3 );set(gca,'FontSize', 16, 'FontName', 'times')
 set(plot1(1), 'LineStyle','-');
 set(plot1(2), 'LineStyle','--');
 set(plot1(3), 'LineStyle','.');
 set(plot1(4), 'LineStyle','-');
 set(plot1(5), 'LineStyle','--');
 set(plot1(6), 'LineStyle','-');
ylim([0, 1]);
xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('CDF of Firm Type, conditional on quintile of Worker Type','FontSize',16, 'FontName', 'times');
%daspect([2 300 1]);
saveas(gcf, 'ed2_H_ycx.eps', 'epsc2');

figure(10)
set(gca,'FontSize', 16, 'FontName', 'times')
daspect([2 3 1]);
plot1 = plot( 100*x, H_xcy(:,1),  100*x, H_xcy(:,round(.2*X.n)),  100*x, H_xcy(:,round(.4*X.n)),  100*x, H_xcy(:,round(0.6*X.n)), 100*x, H_xcy(:,round(0.8*X.n)), 100*x, H_xcy(:,X.n),   'LineWidth',3 );set(gca,'FontSize', 16, 'FontName', 'times')
 set(plot1(1), 'LineStyle','-');
 set(plot1(2), 'LineStyle','--');
 set(plot1(3), 'LineStyle','.');
 set(plot1(4), 'LineStyle','-');
 set(plot1(5), 'LineStyle','--');
 set(plot1(6), 'LineStyle','-');
ylim([0, 1]);
xlabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('CDF of Worker Type, conditional on quintile of Firm Type','FontSize',16, 'FontName', 'times');
%daspect([2 300 1]);
saveas(gcf, 'ed2_H_xcy.eps', 'epsc2');



% figure(401)
% hold on
% h1 = contourf(100*y(65:end),100*x, S_xy(:,65:end), [0 1000000000000] );
% set(gca,'FontSize', 16, 'FontName', 'times')
% axis square
% !daspect([2 3 1]);
% xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
% ylabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
% plot( 100*x(65:end) , xcy_q50(65:end) );
% plot( ycx_q50 , 100*y);
% hold off;
% saveas(gcf, 'ed2_M_xy.eps', 'epsc2');
% 
% 
% figure(301)
% plot( ...
%     exp( 3.034 + 0.966*norminv(xval_x)), ...
%     lognpdf(exp( 3.034 + 0.966*norminv(xval_x)), 3.034, 0.966), ... 
%     exp( 3.034 + 0.092*norminv(yval_y)), ...
%     lognpdf(exp( 3.034 + 0.092*norminv(yval_y)), 3.034, 0.092) )
% xlabel('Worker/Firm Type','FontSize',16, 'FontName', 'times');
% ylabel('Density','FontSize',16, 'FontName', 'times');

STOP
%%



figure(25)
h2 = contourf(100*y,100*x, log(h_xy), log([0.0001:(max(h_xy(:))/25):(0.5737/.88)]) );
set(gca,'FontSize', 16, 'FontName', 'times')
axis square
!daspect([2 3 1]);
xlabel('Firm Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
saveas(gcf, 'ed2_h_xy.eps', 'epsc2');



% transform x and y to log normal
xval_x = exp( 3.034 + 0.966*norminv(xval_x));
yval_y = exp( 3.034 + 0.092*norminv(yval_y));

figure(16)
h1 = contourf(yval_y,xval_x, S_xy, [0 1000000000000] );
set(gca,'FontSize', 16, 'FontName', 'times')
axis square
!daspect([2 3 1]);
xlabel('Firm Type','FontSize',16, 'FontName', 'times');
ylabel('Worker Type','FontSize',16, 'FontName', 'times');
saveas(gcf, 'ed2_M_xy_v2.eps', 'epsc2');

figure(17)
%h2 = contourf(yval_y,xval_x, log(h_xy), log([0.0001:(max(h_xy(:))/25):max(h_xy(:))]) );
h2 = contourf(yval_y,xval_x, (h_xy), ([0.0001:(max(h_xy(:))/25):max(h_xy(:))]) );
set(gca,'FontSize', 16, 'FontName', 'times')
axis square
!daspect([2 3 1]);
xlabel('Firm Type','FontSize',16, 'FontName', 'times');
ylabel('Worker Type','FontSize',16, 'FontName', 'times');
saveas(gcf, 'ed2_h_xy_v2.eps', 'epsc2');


% figure(16)
% contourf(x,y, (h_xy), [(0.0001):((max(h_xy(:)))/25):(max(h_xy(:)))]);
% colorbar

stop



%  
% 
% f_xy_p = importdata('f_xy_planner.dat');
% S_xy_p = importdata('S_xy_planner.dat');
% h_xy_p = importdata('h_xy_planner.dat');
% M_xy_p = importdata('M_xy_planner.dat');
% 
% % figure(11)
% % mesh(f_xy_p)
% % figure(12)
% % mesh(h_xy_p)
% % figure(13)
% % mesh(S_xy_p)
% figure(14)
% contourf(M_xy_p, [0.1 100])
% 


yval_y = linspace(0.001, 0.999, 100)';
xval_x = linspace(0.001, 0.999, 100)';
y_reserve = zeros(100,1);
x_reserve = zeros(100,1);

% Reservation firm type y_reserve(x)
for i=1:100
  tmp =  find( (S_xy(101-i,:) > 0), 1, 'first' ) ;
  if isempty(tmp)
    y_reserve(101-i) = y_reserve(101-i+1);
  else
    y_reserve(101-i) = yval_y(tmp);
  end
end

% Reservation worker type x_reserve(y)
for i=1:100
  tmp =  find( (S_xy(:,101-i) > 0), 1, 'first' ) ;
  if isempty(tmp)
    x_reserve(101-i) = x_reserve(101-i+1);
  else
    x_reserve(101-i) = xval_x(tmp);
  end

end
% 
% % extrapolate reservation types below zero
% beta = regress(x_reserve(65:80), [ones(16,1) xval_x(65:80)]);
% x_reserve_tmp = [ones(100,1) xval_x] * beta;
% x_reserve(1:65) = x_reserve_tmp(1:65);
% plot(x_reserve);
% plot(y_reserve);
% save('data/x_reserve.raw', 'x_reserve',  '-ASCII');
% save('data/y_reserve.raw', 'y_reserve',  '-ASCII');
% 
% 
