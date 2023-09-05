%% Numerical Example: SDOF System with Coulomb Friction
%
% This set-up is based on the Single Degree-of-Freedom Dynamical System
% subjected to Coulomb Friction Force that is presented in the literature:
%
% L. Marino and A. Cicirello (2020). Experimental investigation of a single-
% degree-of-freedom system with Coulomb friction. Nonlinear Dynamics, 99(3), 
% 1781-1799. doi: 10.1007/s11071-019-05443-2
%
% This set-up is applicable only for the Base motion (with fixed wall) case.
%-------------------------------------------------------------------------%
%% Load Numerical Data:
clc; clear;
load('noisy_data');

%% Define key parameters:
omega_n = 3.086;      % Natural frequency of the structure [Hz]
driving_force = 2.5; % Driving force amplitude by the rotor [N]
r_nom = data.frequency_ratio; % Nominal values of the dimensionless input frequency ratios
coulomb_force = flip(driving_force .* data.force_ratio); % Nominal values of the Coulomb Frictions [N]

%% Define the data-set:
Ndata = 15; % Data size per time-step
phase_angle_analytical = zeros(Ndata,1);
phase_angle_analytical(:,1) = data.phase_angle.f010;
phase_angle_analytical(:,2) = data.phase_angle.f025;
phase_angle_analytical(:,3) = data.phase_angle.f040;
phase_angle_analytical(:,4) = data.phase_angle.f055;

% Consolidate the "noisy" Phase angle data:
sigma_phi = 2; % True value of measurement noise for Phase angles
% phase_angle_noisy = phase_angle_analytical + sigma_phi.*randn(Ndata, length(data.force_ratio));
phase_angle_noisy = data.noisy_phase_angles;
data_phase_angle = phase_angle_noisy(2:11,:); % Take only 10 data for Bayesian model updating

% Consolidate the "noisy" input Frequency data:
sigma_r = 0.01; % True value of measurement noise for Frequency ratio
r_v = data.noisy_frequency_ratios; 
frequency_data = r_v .* omega_n;
data_frequency = frequency_data(2:11,:);      % Take only 10 data for Bayesian model updating

% Generate Analytical solution through Den-Hartog's solution:
beta_nom = data.force_ratio;      % Nominal force ratio
output = DenHartogHarmonic(beta_nom');
r_an = output.frequency_ratios;   % The output frequency ratios
phase_an = output.phase_angles;   % The output analytical phase angles
phase_bound = output.phase_bound; % The phase angle bound defined by Den-Hartog's Boundary

%% To plot the Phase angles vs Frequency ratio curves:
colors = [0 0 1; 0 0.5 0; 1 0 0; 1 0 1];

figure; 
hold on; box on; grid on;
for ib = 3 % To plot for different Friction Force ratio
% Numerical scatterplot: 
plot([r_v(:,ib)], phase_angle_noisy(:,ib), 'o', 'color', colors(ib,:), 'linewidth', 2); 
% Analytical plot:
plot([r_an], phase_an(ib+1,:), '--', 'color', colors(ib,:), 'linewidth', 1);
end
plot([r_an], phase_bound,'-- k');
legend(['Data for F_{\mu} = ',num2str(1.0864, '%.3f'), ' N'],...
'Analytical solution', 'Den-Hartog''s Boundary', 'linewidth', 2, 'location', 'southeast');
xlabel('$r$', 'Interpreter', 'latex');  ylabel('$\phi$ $[deg]$', 'Interpreter', 'latex');
xlim([0, 2]); ylim([0 180]); set(gca,'FontSize',20);

%% Bayesian Model Updating Set-up:
% The epistemic parameters to be inferred are the following: 
% {Coulomb Force, Natural Frequency, Frequency Ratio Noise, Phase Angle Noise}

% Define the Prior distribution:
lowerbound = [0.01, 0.001, 0.001]; upperbound = [10, 10, 1];
prior_coulomb = @(x) unifpdf(x, lowerbound(1), upperbound(1));   % Prior for Coulomb Friction
prior_omega = @(x) unifpdf(x, lowerbound(2), upperbound(2));     % Prior for Natural Frequency
prior_sigma_phi = @(x) unifpdf(x, lowerbound(2), upperbound(2)); % Prior for Phase Angle Noise
prior_sigma_r = @(x) unifpdf(x, lowerbound(3), upperbound(3));   % Prior for Frequency Ratio Noise

prior_pdf = @(x) prior_coulomb(x(:,1)) .* prior_omega(x(:,2)) .* ...
                 prior_sigma_phi(x(:,3)) .* prior_sigma_r(x(:,4));
prior_rnd = @(N) [unifrnd(lowerbound(1), upperbound(1), N, 1), ...
                  unifrnd(lowerbound(2), upperbound(2), N, 1), ...
                  unifrnd(lowerbound(2), upperbound(2), N, 1), ...
                  unifrnd(lowerbound(3), upperbound(3), N, 1)];          

% Define the loglikelihood function:
model = @(x,f) blackbox_model(x, f, driving_force);
t = 3;
logL = @(x) loglikelihood(x, model, data_phase_angle(:,t), data_frequency(:,t), r_v(2:11), r_nom(2:11));

%% Define Bayesian Model Updating Parameters:

Nsamples = 1000; % No. of samples to generate from the Posterior
Nbatch = 1;    % No. of sample runs to perform
Ncores = 12;
TMCMC1 = cell(Nbatch,1); TMCMC2 = cell(Nbatch,1); 
timeTMCMC1 = zeros(Nbatch,1); timeTMCMC2 = zeros(Nbatch,1); 

%% Perform Bayesian Model Updating via TMCMC and TMCMC-II:

% Initiate the samplers:

parpool(Ncores)
parfor r = 1:Nbatch

fprintf('Batch no.: %d \n',r)    
    
tic;
TMCMC1{r,1} = TMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,'priorpdf',prior_pdf,...
                           'priorrnd',prior_rnd,'burnin',0,'lastburnin',0);
timeTMCMC1(r,1) = toc;
fprintf('Time elapsed is for the TMCMC sampler: %f \n',timeTMCMC1(r,1))

tic;
TMCMC2{r,1} = TMCMCsampler2('nsamples',Nsamples,'loglikelihood',logL,'priorpdf',prior_pdf,...
                            'priorrnd',prior_rnd,'burnin',0,'lastburnin',0);
timeTMCMC2(r,1) = toc;
fprintf('Time elapsed is for the TMCMC-II sampler: %f \n',timeTMCMC2(r,1))
end
%% Save the data:
save('ESREL2023')

%% 
%{
TMCMC1_cell = cell(50,1); TMCMC2_cell = cell(50,1);
timeTMCMC1_vec = zeros(50,1); timeTMCMC2_vec = zeros(50,1);

for i = 1:22
load('ESREL2023_1.mat', 'TMCMC1', 'TMCMC2', 'timeTMCMC1', 'timeTMCMC2')
TMCMC1_cell{i} = TMCMC1{i}; TMCMC2_cell{i} = TMCMC2{i}; 
timeTMCMC1_vec(i) = timeTMCMC1(i); timeTMCMC2_vec(i) = timeTMCMC2(i); 
end

for i = 23:30
load('ESREL2023_2.mat', 'TMCMC1', 'TMCMC2', 'timeTMCMC1', 'timeTMCMC2')
TMCMC1_cell{i} = TMCMC1{i-22}; TMCMC2_cell{i} = TMCMC2{i-22}; 
timeTMCMC1_vec(i) = timeTMCMC1(i-22); timeTMCMC2_vec(i) = timeTMCMC2(i-22);
end

for i = 31:40
load('ESREL2023_3.mat', 'TMCMC1', 'TMCMC2', 'timeTMCMC1', 'timeTMCMC2')
TMCMC1_cell{i} = TMCMC1{i-30}; TMCMC2_cell{i} = TMCMC2{i-30}; 
timeTMCMC1_vec(i) = timeTMCMC1(i-30); timeTMCMC2_vec(i) = timeTMCMC2(i-30); 
end

for i = 41:50
load('ESREL2023_4.mat', 'TMCMC1', 'TMCMC2', 'timeTMCMC1', 'timeTMCMC2')
TMCMC1_cell{i} = TMCMC1{i-40}; TMCMC2_cell{i} = TMCMC2{i-40}; 
timeTMCMC1_vec(i) = timeTMCMC1(i-40); timeTMCMC2_vec(i) = timeTMCMC2(i-40); 
end

TMCMC1 = TMCMC1_cell; TMCMC2 = TMCMC2_cell; timeTMCMC1 = timeTMCMC1_vec; timeTMCMC2 = timeTMCMC2_vec;
save('ESREL2023.mat', 'TMCMC1', 'TMCMC2', 'timeTMCMC1', 'timeTMCMC2')
%}

%% P-box Analysis:
load('ESREL2023')

sampsTMCMC1 = zeros(Nsamples, length(TMCMC1),2); sampsTMCMC2 = zeros(Nsamples, length(TMCMC2),2);
pboxTMCMC1 = zeros(Nsamples,2,2); pboxTMCMC2 = zeros(Nsamples,2,2);
for i = 1:length(TMCMC1)
cell1 = TMCMC1{i}; cell2 = TMCMC2{i};
samps1 = cell1.samples; samps2 = cell2.samples;
sampsTMCMC1(:,i,1) = sort(samps1(:,1), 'ascend'); sampsTMCMC1(:,i,2) = sort(samps1(:,2), 'ascend');
sampsTMCMC2(:,i,1) = sort(samps2(:,1), 'ascend'); sampsTMCMC2(:,i,2) = sort(samps2(:,2), 'ascend');
end

for j = 1:Nsamples
pboxTMCMC1(j,:,1) = [min(sampsTMCMC1(j,:,1)),max(sampsTMCMC1(j,:,1))]; pboxTMCMC1(j,:,2) = [min(sampsTMCMC1(j,:,2)),max(sampsTMCMC1(j,:,2))]; 
pboxTMCMC2(j,:,1) = [min(sampsTMCMC2(j,:,1)),max(sampsTMCMC2(j,:,1))]; pboxTMCMC2(j,:,2) = [min(sampsTMCMC2(j,:,2)),max(sampsTMCMC2(j,:,2))];
end

% Pbox of estimates:
figure;
subplot(1,2,1)
hold on; box on; grid on;
[f1,x1] = ecdf(pboxTMCMC1(:,1,1)); [f2,x2] = ecdf(pboxTMCMC1(:,2,1));
stairs(x1,f1, 'b', 'linewidth', 2); stairs(x2,f2, 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');

[f1,x1] = ecdf(pboxTMCMC2(:,1,1)); [f2,x2] = ecdf(pboxTMCMC2(:,2,1));
stairs(x1,f1, 'r', 'linewidth', 2); stairs(x2,f2, 'r', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'r', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'r', 'linewidth', 2, 'handlevisibility', 'off');
xline(1.0864, 'k--', 'linewidth', 2);
legend('P-box TMCMC', 'P-box TMCMC-II', 'True value F_{\mu} = 1.086 [N]', 'linewidth', 2)
xlabel('$F_{\mu}$ $[N]$', 'Interpreter', 'latex'); ylabel('ECDF value'); set(gca, 'Fontsize', 18)
xlim([0.7, 1.8])

subplot(1,2,2)
hold on; box on; grid on;
[f1,x1] = ecdf(pboxTMCMC1(:,1,2)); [f2,x2] = ecdf(pboxTMCMC1(:,2,2));
stairs(x1,f1, 'b', 'linewidth', 2); stairs(x2,f2, 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');

[f1,x1] = ecdf(pboxTMCMC2(:,1,2)); [f2,x2] = ecdf(pboxTMCMC2(:,2,2));
stairs(x1,f1, 'r', 'linewidth', 2); stairs(x2,f2, 'r', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'r', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'r', 'linewidth', 2, 'handlevisibility', 'off');
xline(3.086, 'k--', 'linewidth', 2);
legend('P-box TMCMC', 'P-box TMCMC-II', 'True value \omega_n = 3.086 [Hz]', 'linewidth', 2)
xlabel('$\omega_n$ $[Hz]$', 'Interpreter', 'latex'); ylabel('ECDF value'); set(gca, 'Fontsize', 18)

area_mat = zeros(2,2);
area_mat(1,1) = areaMe(pboxTMCMC1(:,1,1),pboxTMCMC1(:,2,1)); area_mat(1,2) = areaMe(pboxTMCMC1(:,1,2),pboxTMCMC1(:,2,2));
area_mat(2,1) = areaMe(pboxTMCMC2(:,1,1),pboxTMCMC2(:,2,1)); area_mat(2,2) = areaMe(pboxTMCMC2(:,1,2),pboxTMCMC2(:,2,2));
T = array2table(area_mat,'VariableNames', ...
                {'Coulomb_Friction_Pbox_area', 'Natural_Frequency_Pbox_area'},...
                'RowNames', {'TMCMC', 'TMCMC-II'});

area_vec = [0.081508, 0.17135; 0.008878, 0.0089148];
x = categorical({'F_{\mu} [N]', '\omega_{n} [Hz]'}); x = reordercats(x,{'F_{\mu} [N]', '\omega_{n} [Hz]'});

figure;
hold on; box on; grid on;
b = bar(x, area_vec);
xlabel('Parameter'); ylabel('Area of P-box'); 
set(gca, 'Fontsize', 18); legend('TMCMC', 'TMCMC-II', 'linewidth', 2)          

%% Mean Analysis:

meanTMCMC1 = zeros(length(TMCMC1),2); meanTMCMC2 = zeros(length(TMCMC2),2);
betaTMCMC1 = zeros(length(TMCMC1),1); betaTMCMC2 = zeros(length(TMCMC2),1);

for i = 1:length(TMCMC1)
cell1 = TMCMC1{i}; cell2 = TMCMC2{i};
samps1 = cell1.samples; samps2 = cell2.samples;
meanTMCMC1(i,:) = mean(samps1(:,1:2)); meanTMCMC2(i,:) = mean(samps2(:,1:2)); 
betaTMCMC1(i) = length(cell1.beta)-1; betaTMCMC2(i) = length(cell2.beta)-1; 
end

figure; 
subplot(2,2,1)
hold on; box on; grid on;
histogram(meanTMCMC1(:,1))
xlabel('$F_{\mu}$ $[N]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
title('Mean values TMCMC')
subplot(2,2,2)
hold on; box on; grid on;
histogram(meanTMCMC1(:,2))
xlabel('$\omega_n$ $[Hz]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
title('Mean values TMCMC')

subplot(2,2,3)
hold on; box on; grid on;
histogram(meanTMCMC2(:,1))
xlabel('$F_{\mu}$ $[N]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
title('Mean values TMCMC2')
subplot(2,2,4)
hold on; box on; grid on;
histogram(meanTMCMC2(:,2))
xlabel('$\omega_n$ $[Hz]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
title('Mean values TMCMC2')

figure;
hold on; box on; grid on;
[f1,x1] = ecdf(betaTMCMC1); [f2,x2] = ecdf(betaTMCMC2);
stairs(x1,f1, 'b', 'linewidth', 2); stairs(x2,f2, 'r', 'linewidth', 2); 
plot([11,12], [1,1], 'b', 'linewidth',2, 'handlevisibility', 'off')
xticks([10:12]); xlabel('No. of Transition steps'); ylabel('ECDF values'); set(gca, 'Fontsize', 18)
 
beta_vec = [6,5 ; 44,36 ; 0,9];
iterations = [9, 10, 11];

figure;
hold on; box on; grid on;
b = bar(iterations, beta_vec);
xlabel('No. of Iterations'); ylabel('Count'); xticks([9:11])
set(gca, 'Fontsize', 18); legend('TMCMC', 'TMCMC-II', 'linewidth', 2)

