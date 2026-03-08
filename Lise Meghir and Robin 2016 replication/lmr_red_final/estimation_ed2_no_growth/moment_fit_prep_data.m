
% read in cohort simulation

firm_id = importdata('panel_firm_id.dat');
firm_type = importdata('panel_firm_type.dat');
empl = importdata('panel_status.dat');
wage = importdata('panel_wage.dat');

% First save the data for use in Stata (to look at covariance structure in
% an easier way
month = (1:size(firm_id,2));
month = repmat( month, size(firm_id,1), 1);
worker_id = (1:size(firm_id,1))';
worker_id = repmat(worker_id, 1, size(firm_id,2));
year = ceil(month/12);

worker_id_long = reshape(worker_id, size(worker_id,1)*size(worker_id,2), 1);
month_long = reshape(month, size(worker_id,1)*size(worker_id,2), 1);
year_long = reshape(year, size(worker_id,1)*size(worker_id,2), 1);
firm_id_long = reshape(firm_id, size(worker_id,1)*size(worker_id,2), 1);
firm_type_long = reshape(firm_type, size(worker_id,1)*size(worker_id,2), 1);
wage_long = reshape(wage, size(worker_id,1)*size(worker_id,2), 1);
empl_long = reshape(empl, size(worker_id,1)*size(worker_id,2), 1);

tmp_data = [worker_id_long, month_long, year_long, firm_id_long, firm_type_long, wage_long, empl_long];

save wages_for_stata_to_compute_moments_ed1.raw tmp_data -ASCII -tabs
