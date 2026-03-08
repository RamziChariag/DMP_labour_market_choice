% create matrix of derivatives

tmp = importdata('data/ident_theta_10001.dat');
theta   = zeros(size(tmp,1),size(tmp,2),15);

for i=1:15
	theta(:,:,i) = importdata(['data/ident_theta_',num2str(10000+i),'.dat']);
end

ident2 = zeros(size(theta,1),size(theta,3));

for p=1:size(theta,3)
  for m=1:size(theta,1)
      index = find(theta(2,:,p)>min(theta(2,:,p)) & ~isnan(theta(2,:,p)) & ~isnan(theta(m,:,p)));
      b = regress( (theta(m,index,p)')/mean(theta(m,index,p)')-1,  (theta(1,index,p)')/mean(theta(1,index,p)')-1);
      ident2(m,p) = b;
  end
end

ident2(1:2,:) = [];

csvwrite('data/local_identification_ed2.csv', ident2);

