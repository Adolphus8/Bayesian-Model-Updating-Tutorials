%% Toy Problem: 2D Coupled spring-mass system
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Reference: http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/
% node100.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% We have a coupled oscillator configuration: spring > mass > spring > mass >
% spring.
% 
% Eigenfrequencies: sqrt(k./m), sqrt((k + 2.*k_12)./m)
% Hence, theoretical eigenfrequencies = 1.0954 Hz, 2.2804 Hz
%
% Input data: 
% Primary spring stiffness, k = 0.6 N/m; 
% Secondary spring stiffness, k_12 = 1 N/m; 
% Mass, m = 0.5 kg
%
% In this online Bayesian Model Updating problem,we assume we have a stream
% of measurements coming at every arbitrary time-step. This stream of
% consists of 15 inidividual measurements with different number of
% measurements obtained at each time-step. In this problem, 15 measurements
% are obtained over the course of 5 time-steps and are distributed as:
% [3,4,5,1,2].
%
% Here, Bayesian Model Updating is performed via the SEMC sampler.
%
%% Define the parameters and random variables:

m = 0.5;  % Mass of the blocks in [kg]
k = 0.6;  % Stiffness of primary spring [N/m]
k_12 = 1; % Stiffness of secondary spring [N/m]

%% Define the model:

% Define model for the first eigenfrequency:
model_1 = @(x) sqrt(x(:,2)./x(:,1));

% Define model for the second eigenfrequency:
model_2 = @(x) sqrt((x(:,2) + 2.*x(:,3))./x(:,1));

%% Generate noisy measurements of Eigenfrequencies:

% Define the stochastic noise term for eigenfrequency 1:
noise_1 = 0.1*model_1([m,k])*randn(15,1);

% Define the stochastic noise term for eigenfrequency 2:
noise_2 = 0.1*model_2([m,k,k_12])*randn(15,1);

% Define the "noisy" measurements:
measurements = [model_1([m,k]), model_2([m,k,k_12])] + [noise_1, noise_2];

% To plot the 2D scatter plot of the measurements:
figure;
hold on; box on; grid on
scatter(measurements(:,1), measurements(:,2), 10, 'r', 'filled');
plot(model_1([m,k]), model_2([m,k,k_12]), 'k +','LineWidth', 2);
xlabel('\omega_1^{noisy} [Hz]')
ylabel('\omega_2^{noisy} [Hz]')
legend('Noisy eigenfrequencies', 'True eigenfrequency','LineWidth',2)
set(gca, 'fontsize', 15)
ylim([1.5 2.8])
hold off

%% Define the Prior:

lowerBound = [0.01, 1e-05]; upperBound = [4, 1]; 

% Prior PDF of k: 
priorPDF_k = @(x) unifpdf(x, lowerBound(1), upperBound(1)); 

% Prior PDF of k_12: 
priorPDF_k12 = @(x) unifpdf(x, lowerBound(1), upperBound(1)); 

% Prior PDF of sigma_1 (standard deviation of f1): 
priorPDF_sigma1 = @(x) unifpdf(x, lowerBound(2), upperBound(2)); 

% Prior PDF of sigma_2 (standard deviation of f2): 
priorPDF_sigma2 = @(x) unifpdf(x, lowerBound(2), upperBound(2)); 

% Define the overall prior PDF:
prior_pdf = @(x) priorPDF_k(x(:,1)).*priorPDF_k12(x(:,2)).*...
                 priorPDF_sigma1(x(:,3)).*priorPDF_sigma2(x(:,4)); 

prior_rnd = @(N) [unifrnd(lowerBound(1), upperBound(1), N, 1),...
                  unifrnd(lowerBound(1), upperBound(1), N, 1),...
                  unifrnd(lowerBound(2), upperBound(2), N, 1),...
                  unifrnd(lowerBound(2), upperBound(2), N, 1)]; 
              
%% Define the Log-likelihood function:
% x: vector of epistemic parameters;
% mea: measurement vector for the eigenfrequencies;
% mod1: model output from model_1;
% mod2: model output from model_2;

logL = @(x, measurements) - 0.5 .* (1./x(:,3)).^2 .*(measurements(:,1) - model_1([m,x(:,1)]))' *...
                                      (measurements(:,1) - model_1([m,x(:,1)])) -...
                                       length(measurements).*log(sqrt(2*pi).*x(:,3)) +...
                          - 0.5 .* (1./x(:,4)).^2 .*(measurements(:,2) - model_2([m, x(:,1), x(:,2)]))' *...
                                      (measurements(:,2) - model_2([m, x(:,1), x(:,2)])) -...
                                       length(measurements).*log(sqrt(2*pi).*x(:,4));

%% Define the likelihood function cell array
% This is to indicate measurements obtained separately at different
% time-step.

% 15 measurements are obtained over the course of 5 time-steps and are 
% distributed according to: [3,4,5,1,2].

% To create a cell array of Likelihood functions:
logl{1} = @(x) logL(x, measurements(1:3,:));
logl{2} = @(x) logL(x, measurements(4:7,:));
logl{3} = @(x) logL(x, measurements(8:12,:));
logl{4} = @(x) logL(x, measurements(13,:));
logl{5} = @(x) logL(x, measurements(14:end,:));

%% Perform Online Sequential Bayesian Updating:

% Initialise:
Nsamples = 1000;

% Start SEMC sampler:
tic;
SEMC = SEMCsampler('nsamples',Nsamples,'loglikelihoods',logl,...
               'priorpdf',prior_pdf,'priorrnd',prior_rnd,'burnin',0);
semc_samples = SEMC.samples;
semc_allsamples = SEMC.allsamples;
timeSEMC = toc;
fprintf('Time elapsed is for the SEMC sampler: %f \n',timeSEMC)

%% Perform Offline Bayesian Updating via TMCMC:

% To create a cell array of Likelihood functions:
loglike{1} = @(x) logL(x, measurements(1:3,:));
loglike{2} = @(x) logL(x, measurements(1:7,:));
loglike{3} = @(x) logL(x, measurements(1:12,:));
loglike{4} = @(x) logL(x, measurements(1:13,:));
loglike{5} = @(x) logL(x, measurements(1:15,:));

for i = 1:length(logl)
    
fprintf('Commence Offline BMU iteration t = %d \n', i)
                                   
% Start TMCMC sampler:
tic;
TMCMC{i} = TMCMCsampler('nsamples',Nsamples,'loglikelihood',loglike{i},...
                   'priorpdf',prior_pdf,'priorrnd',prior_rnd,...
                   'burnin',0);
tmcmc_allsamples(:,:,i) = TMCMC{i}.samples;
timeTMCMC(i) = toc;
fprintf('Time elapsed is for the TMCMC sampler: %f \n',timeTMCMC(i))                                  

end

%% Plot the combined Scatterplot matrix:

Posterior_SEMC = SEMC.allsamples;

for t = 1:length(logl)+1
figure();
[~,ax1] = plotmatrix(Posterior_SEMC(:,:,t));
for i=1:4
    ax1(i,1).FontSize = 16; 
    ax1(4,i).FontSize = 16; 
end
ax1(1,1).YLim = [-0.3, 2.7]; ax1(2,1).YLim = [-0.3, 2.3]; 
ax1(3,1).YLim = [-0.15, 1.15]; ax1(4,1).YLim = [-0.15,1.15]; 
ax1(4,1).XLim = [-0.3, 2.7]; ax1(4,2).XLim = [-0.3, 2.3]; 
ax1(4,3).XLim = [-0.15, 1.15]; ax1(4,4).XLim = [-0.15,1.15]; 

ylabel(ax1(1,1),'k [N/m]'); ylabel(ax1(2,1),'k_{12} [N/m]');
ylabel(ax1(3,1),'\sigma_1 [Hz]'); ylabel(ax1(4,1),'\sigma_2 [Hz]');
xlabel(ax1(4,1),'k [N/m]'); xlabel(ax1(4,2),'k_{12} [N/m]');
xlabel(ax1(4,3),'\sigma_1 [Hz]'); xlabel(ax1(4,4),'\sigma_2 [Hz]');
title(sprintf('SEMC Posterior at Time-step = %2d \n', t-1))
set(gca,'FontSize',16)

end

%% Analysis of samples:

posterior_mean_semc = zeros(size(logl,2),size(semc_allsamples,2));
posterior_std_semc = zeros(size(logl,2),size(semc_allsamples,2));
posterior_bounds_k_semc = zeros(size(logl,2),2);
posterior_bounds_k12_semc = zeros(size(logl,2),2);
posterior_bounds_sigma1_semc = zeros(size(logl,2),2);
posterior_bounds_sigma2_semc = zeros(size(logl,2),2);

for idx = 1:size(logl,2)
posterior_mean_semc(idx,1) = mean(semc_allsamples(:,1,idx+1));
posterior_mean_semc(idx,2) = mean(semc_allsamples(:,2,idx+1));
posterior_mean_semc(idx,3) = mean(semc_allsamples(:,3,idx+1));
posterior_mean_semc(idx,4) = mean(semc_allsamples(:,4,idx+1));

posterior_std_semc(idx,1) = std(semc_allsamples(:,1,idx+1));
posterior_std_semc(idx,2) = std(semc_allsamples(:,2,idx+1));
posterior_std_semc(idx,3) = std(semc_allsamples(:,3,idx+1));
posterior_std_semc(idx,4) = std(semc_allsamples(:,4,idx+1));

posterior_bounds_k_semc(idx,:) = prctile(semc_allsamples(:,1,idx+1), [5, 95]);
posterior_bounds_k12_semc(idx,:) = prctile(semc_allsamples(:,2,idx+1), [5, 95]);
posterior_bounds_sigma1_semc(idx,:) = prctile(semc_allsamples(:,3,idx+1), [5, 95]);
posterior_bounds_sigma2_semc(idx,:) = prctile(semc_allsamples(:,4,idx+1), [5, 95]);
end
posterior_cov_semc = (posterior_std_semc./posterior_mean_semc).*100;

posterior_mean_tmcmc = zeros(size(logl,2),size(tmcmc_allsamples,2));
posterior_std_tmcmc = zeros(size(logl,2),size(tmcmc_allsamples,2));
posterior_bounds_k_tmcmc = zeros(size(logl,2),2);
posterior_bounds_k12_tmcmc = zeros(size(logl,2),2);
posterior_bounds_sigma1_tmcmc = zeros(size(logl,2),2);
posterior_bounds_sigma2_tmcmc = zeros(size(logl,2),2);

for idx = 1:size(logl,2)
posterior_mean_tmcmc(idx,1) = mean(tmcmc_allsamples(:,1,idx));
posterior_mean_tmcmc(idx,2) = mean(tmcmc_allsamples(:,2,idx));
posterior_mean_tmcmc(idx,3) = mean(tmcmc_allsamples(:,3,idx));
posterior_mean_tmcmc(idx,4) = mean(tmcmc_allsamples(:,4,idx));

posterior_std_tmcmc(idx,1) = std(tmcmc_allsamples(:,1,idx));
posterior_std_tmcmc(idx,2) = std(tmcmc_allsamples(:,2,idx));
posterior_std_tmcmc(idx,3) = std(tmcmc_allsamples(:,3,idx));
posterior_std_tmcmc(idx,4) = std(tmcmc_allsamples(:,4,idx));

posterior_bounds_k_tmcmc(idx,:) = prctile(tmcmc_allsamples(:,1,idx), [5, 95]);
posterior_bounds_k12_tmcmc(idx,:) = prctile(tmcmc_allsamples(:,2,idx), [5, 95]);
posterior_bounds_sigma1_tmcmc(idx,:) = prctile(tmcmc_allsamples(:,3,idx), [5, 95]);
posterior_bounds_sigma2_tmcmc(idx,:) = prctile(tmcmc_allsamples(:,4,idx), [5, 95]);
end
posterior_cov_tmcmc = (posterior_std_tmcmc./posterior_mean_tmcmc).*100;

%% Analysis of the model updating:

figure;
subplot(1,2,1)
hold on; grid on; box on;
plot([0 size(semc_allsamples,3)], [0.6 0.6], 'k--', 'linewidth', 1)
y_neg_a1i = abs(posterior_mean_tmcmc(:,1) - posterior_bounds_k_tmcmc(:,1)); % error in the negative y-direction
y_pos_a1i = abs(posterior_mean_tmcmc(:,1) - posterior_bounds_k_tmcmc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_tmcmc,1))', posterior_mean_tmcmc(:,1), y_neg_a1i, y_pos_a1i, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
y_neg_a1ii = abs(posterior_mean_semc(:,1) - posterior_bounds_k_semc(:,1)); % error in the negative y-direction
y_pos_a1ii = abs(posterior_mean_semc(:,1) - posterior_bounds_k_semc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_semc,1))', posterior_mean_semc(:,1), y_neg_a1ii, y_pos_a1ii, '-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red', 'linewidth',1);
legend('True value', 'TMCMC estimated values', 'SEMC estimated values', 'linewidth', 2)
xlim([0 size(semc_allsamples,3)])
xlabel('Iteration, j')
ylabel('Primary stiffness, k [N/m]')
set(gca, 'fontsize', 18)

subplot(1,2,2)
hold on; grid on; box on;
plot([0 size(semc_allsamples,3)], [1.0 1.0], 'k--', 'linewidth', 1)
y_neg_b1i = abs(posterior_mean_tmcmc(:,2) - posterior_bounds_k12_tmcmc(:,1)); % error in the negative y-direction
y_pos_b1i = abs(posterior_mean_tmcmc(:,2) - posterior_bounds_k12_tmcmc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_tmcmc,1))', posterior_mean_tmcmc(:,2), y_neg_b1i, y_pos_b1i, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
y_neg_b1ii = abs(posterior_mean_semc(:,2) - posterior_bounds_k12_semc(:,1)); % error in the negative y-direction
y_pos_b1ii = abs(posterior_mean_semc(:,2) - posterior_bounds_k12_semc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_semc,1))', posterior_mean_semc(:,2), y_neg_b1ii, y_pos_b1ii, '-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red', 'linewidth',1);
legend('True value', 'TMCMC estimated values', 'SEMC estimated values', 'linewidth', 2)
xlim([0 size(semc_allsamples,3)])
xlabel('Iteration, j')
ylabel('Secondary stiffness, k_{12} [N/m]')
set(gca, 'fontsize', 18)

figure;
subplot(1,2,1)
hold on; grid on; box on;
plot([0 size(semc_allsamples,3)], [0.1*model_1([m,k]) 0.1*model_1([m,k])], 'k--', 'linewidth', 1)
y_neg_a2i = abs(posterior_mean_tmcmc(:,3) - posterior_bounds_sigma1_tmcmc(:,1)); % error in the negative y-direction
y_pos_a2i = abs(posterior_mean_tmcmc(:,3) - posterior_bounds_sigma1_tmcmc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_tmcmc,1))', posterior_mean_tmcmc(:,3), y_neg_a2i, y_pos_a2i, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
y_neg_a2ii = abs(posterior_mean_semc(:,3) - posterior_bounds_sigma1_semc(:,1)); % error in the negative y-direction
y_pos_a2ii = abs(posterior_mean_semc(:,3) - posterior_bounds_sigma1_semc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_semc,1))', posterior_mean_semc(:,3), y_neg_a2ii, y_pos_a2ii, '-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red', 'linewidth',1);
legend('True value', 'TMCMC estimated values', 'SEMC estimated values', 'linewidth', 2)
xlim([0 size(semc_allsamples,3)])
xlabel('Iteration, j')
ylabel('Noise standard deviation 1, \sigma_{1} [Hz]')
set(gca, 'fontsize', 18)

subplot(1,2,2)
hold on; grid on; box on;
plot([0 size(semc_allsamples,3)], [0.1*model_2([m,k,k_12]) 0.1*model_2([m,k,k_12])], 'k--', 'linewidth', 1)
y_neg_b2i = abs(posterior_mean_tmcmc(:,4) - posterior_bounds_sigma2_tmcmc(:,1)); % error in the negative y-direction
y_pos_b2i = abs(posterior_mean_tmcmc(:,4) - posterior_bounds_sigma2_tmcmc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_tmcmc,1))', posterior_mean_tmcmc(:,4), y_neg_b2i, y_pos_b2i, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
y_neg_b2ii = abs(posterior_mean_semc(:,4) - posterior_bounds_sigma2_semc(:,1)); % error in the negative y-direction
y_pos_b2ii = abs(posterior_mean_semc(:,4) - posterior_bounds_sigma2_semc(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean_semc,1))', posterior_mean_semc(:,4), y_neg_b2ii, y_pos_b2ii, '-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red', 'linewidth',1);
legend('True value', 'TMCMC estimated values', 'SEMC estimated values', 'linewidth', 2)
xlim([0 size(semc_allsamples,3)])
xlabel('Iteration, j')
ylabel('Noise standard deviation 2, \sigma_{2} [Hz]')
set(gca, 'fontsize', 18)

%% Model Update

update_model_1 = @(x) sqrt(x./m);
update_model_2 = @(x) sqrt((x(:,1) + 2.*x(:,2))./m);

figure;
subplot(1,2,1)
hold on; box on; grid on
scatter(update_model_1(tmcmc_allsamples(:,1,5)),...
update_model_2([tmcmc_allsamples(:,1,5),tmcmc_allsamples(:,2,5)]), 10, 'b', 'filled')
scatter(measurements(:,1), measurements(:,2), 10, 'r', 'filled');
plot(model_1([m,k]), model_2([m,k,k_12]), 'k +','LineWidth', 2);
xlabel('\omega_1^{noisy} [Hz]')
ylabel('\omega_2^{noisy} [Hz]')
legend('TMCMC Model Update','Noisy eigenfrequencies', 'True eigenfrequency','LineWidth',2)
set(gca, 'fontsize', 15)

subplot(1,2,2)
hold on; box on; grid on
scatter(update_model_1(semc_samples(:,1)),...
update_model_2([semc_samples(:,1),semc_samples(:,2)]), 10, 'b', 'filled')
scatter(measurements(:,1), measurements(:,2), 10, 'r', 'filled');
plot(model_1([m,k]), model_2([m,k,k_12]), 'k +','LineWidth', 2);
xlabel('\omega_1^{noisy} [Hz]')
ylabel('\omega_2^{noisy} [Hz]')
legend('SEMC Model Update','Noisy eigenfrequencies', 'True eigenfrequency','LineWidth',2)
set(gca, 'fontsize', 15)

%% SEMC Statistics

dim = size(semc_samples,2); % dimensionality of the problem
target_accept = 0.23 + (0.21./dim);

% Plot the acceptance rate values across iterations:
figure;
hold on; box on; grid on;
plot([1 size(semc_allsamples,3)],[target_accept target_accept] , 'c','linewidth', 1.5)
plot([1 size(semc_allsamples,3)],[0.15 0.15] , 'k','linewidth', 1.5)
plot((1:size(semc_allsamples,3)-1)', SEMC.acceptance, '--rs', 'MarkerFaceColor','r','linewidth', 1.5)
plot([1 size(semc_allsamples,3)],[0.5 0.5] , 'k','linewidth', 1.5)
legend('Target acceptance rate', 'Optimum acceptance limits', 'SEMC acceptance rates', 'linewidth', 2)
title('Plot of Acceptance rates')
xlabel('Iteration number, j')
ylabel('Acceptance rate')
xlim([1 6])
ylim([0.1 0.7])
set(gca, 'fontsize', 18)

%% Save the data:

save('example_CoupledOscillator_SEMC_m');
