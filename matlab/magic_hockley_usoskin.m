%% MAGIC Pre-processing for Hockley and Usoskin datasets

% go to correct directory before you start!
cd 'C:/Users/Bettina/Desktop/MAGIC-master/matlab';

%% load Jims data
% data should be cells as rows and genes as columns
file = 'C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Visceraltpm_m_fltd_mat.tsv'; 
data = importdata(file);
data = data';

%% MAGIC
[pc_imputed, U, ~] = run_magic(data, 'npca', 100, 'k', 15, 'a', 15, 'make_plot_opt_t', false);
magic_data = pc_imputed * U';
magic_data = magic_data';
magic_data(magic_data<0) = 0;

% Save results
dlmwrite('jim_data_magic.tsv', magic_data, 'delimiter', '\t');

%% load Usoskin data
file = 'C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/usoskin_m_fltd_mat.tsv'; 
data = importdata(file);
data = data';

%% MAGIC
[pc_imputed, U, ~] = run_magic(data, 'npca', 100, 'k', 15, 'a', 15, 'make_plot_opt_t', false);
magic_data = pc_imputed * U';
magic_data = magic_data';
magic_data(magic_data<0) = 0;

% Save results
dlmwrite('usoskin_data_magic.tsv', magic_data, 'delimiter', '\t');
