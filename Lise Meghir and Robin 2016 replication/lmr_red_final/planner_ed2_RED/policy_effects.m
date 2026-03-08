clear 
figure(1)
W0 = importdata('W0_all.dat');

x= 100*linspace(0,1,size(W0,1))';

for i=1:6
   W0_diff(:,i) = ((W0(:,i)) ) ./ W0(:,1) ;
end
! plot(x, W0_diff(:,1), x, W0_diff(:,2), x, W0_diff(:,4), x, W0_diff(:,5) , x, W0_diff(:,6) );
plot(x, W0_diff(:,1), x, W0_diff(:,2), x, W0_diff(:,4), x, W0_diff(:,5)  );
set(gca,'FontSize', 16, 'FontName', 'times')
axis([0 100 0.98 1.05]);
legend('Decentralized', 'Planner', 'Minimum Wage', 'Severence Pay', 'Unemployment Insurance')
xlabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('W_0(x) Relative to Decentralized Economy','FontSize',16, 'FontName', 'times');
saveas(gcf, 'W0_policy.eps', 'epsc2');


figure(2)
u_x = importdata('u_all.dat');
! plot(x, u_x(:,1), x, u_x(:,2), x, u_x(:,4), x, u_x(:,5) , x, u_x(:,6) );
plot(x, u_x(:,1), x, u_x(:,2), x, u_x(:,4), x, u_x(:,5)  );

set(gca,'FontSize', 16, 'FontName', 'times')
axis([0 100 0 1]);
legend('Decentralized', 'Planner', 'Minimum Wage', 'Severence Pay', 'Unemployment Insurance')
xlabel('Worker Type (Percentile)','FontSize',16, 'FontName', 'times');
ylabel('Unemployment Rate (Percent)','FontSize',16, 'FontName', 'times');
saveas(gcf, 'u_policy.eps', 'epsc2');

