% read in bootstrapped moments from STATA output and create standard
% deviaitons to use in graphs

moments_ed1 = zeros(100,241);
moments_ed2 = zeros(100,241);

for i=1:100
 moments_ed1(i,:) = importdata( sprintf('data/bs_mom_ed1_%d.raw', i) );
 moments_ed2(i,:) = importdata( sprintf('data/bs_mom_ed2_%d.raw', i) );
end

std_ed1 = std(moments_ed1);
std_ed2 = std(moments_ed2);

dlmwrite('std_ed1_bs.raw',std_ed1,'delimiter','\n');
dlmwrite('std_ed2_bs.raw',std_ed2,'delimiter','\n');

