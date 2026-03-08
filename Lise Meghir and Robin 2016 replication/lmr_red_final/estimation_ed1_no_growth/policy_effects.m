clear 
figure(1)
W0_ed1 = importdata('../estimation_ed1_no_growth/W0_x.dat');
W0_UI_ed1 = importdata('../policy_UI_ed1_RED/W0_x.dat');

W0_ed2 = importdata('../estimation_ed2_no_growth/W0_x.dat');
W0_UI_ed2 = importdata('../policy_UI_ed2_RED/W0_x.dat');

W_gain_ed1 = (W0_UI_ed1-W0_ed1)./W0_ed1;
W_gain_ed2 = (W0_UI_ed2-W0_ed2)./W0_ed2;

% smooth out some minor numerical noise
W_gain_ed2(2:end-1) = ( W_gain_ed2(1:end-2) + W_gain_ed2(2:end-1)  + W_gain_ed2(3:end) ) /3;

x= 100*linspace(0,1,size(W0_ed1,1))';

plot1 = plot(x, W_gain_ed1, x, W_gain_ed2, 'linewidth', 3 );
set(gca,'FontSize', 16, 'FontName', 'times');
set(plot1(2), 'LineStyle','--');
grid on;
axis([0 100 -0.015 0.045]);
xlabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('Relative difference in value of unemployment','FontSize',16, 'FontName', 'times');
legend('High school or less', 'College graduate', 'Location', 'southwest');
%legend('boxoff');
saveas(gcf, 'W0_policy.eps', 'epsc2');
