%% Toy problem: 1D Linear Static Spring-Mass system
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Problem: We have a simple linear static spring-mass system whose spring 
% has a constant stiffness, k. With a known displacement, we want to
% estimate k using Online Bayesian Model Updating where we have
% measurements obtained at different time-steps.
%
% True value of k = 263 N/m 
% (This is the spring constant for those used in pens) 
%
% In this online Bayesian Model Updating problem,we assume we have a stream
% of measurements coming at every arbitrary time-step. This stream of
% consists of 15 inidividual measurements whereby one measurement is 
% obtained at each time-step. This code demonstrates not only the 
% Sequential Bayesian Model Updating with every time-step for each
% measurement, but also illustrates how the posterior for each epistemic
% parameter changes with increased measurements. 
%
% Here, Bayesian Model Updating is performed via SEMC sampler.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1) Defining the variables:

Nmeasurements = 15;
displacement = unifrnd(0.02,0.08,Nmeasurements,1);        % displacement [m]
displacement = sort(displacement);                        % sort displacement in ascending order
stiffness = 263;                                          % True value of k [N/m] 

%% 2) Defining the model:

model = @(k,d) - k.*d;

noise_sd = 1;
measurements = model(stiffness,displacement) + noise_sd*randn(Nmeasurements ,1);

% To plot the measurements:
figure;
hold on; box on; grid on;
scatter(displacement, measurements, 13, 'r', 'filled')
disp = linspace(0.02,0.08,50);
plot(disp, model(stiffness,disp), 'k --', 'LineWidth', 1)
legend('Noisy measurements','True measurements','LineWidth',2)
xlim([0.02 0.08])
xlabel('Displacement, d [m]')
ylabel('Force, F [N]')
set(gca, 'Fontsize', 16)

%% Define the Prior distribution:

prior_k = @(x) unifpdf(x,1,1000);       % Defining the Prior distribution for k.
prior_sigma = @(x) unifpdf(x,1e-05,2);  % Defining the Prior distribution for sigma.
prior_pdf = @(x) prior_k(x(:,1)).*prior_sigma(x(:,2));

% To draw random samples from the Prior:
prior_rnd = @(N) [unifrnd(1,1000,N,1), unifrnd(1e-05,2,N,1)]; 

%% Define the likelihood function cell array
% This is to indicate measurements obtained separately at different
% time-step.

time_step = 15;

% To create array of Likelihood functions:
logL = cell(time_step,1);
for i = 1:time_step
idx = i; % index variable
logL{i} = @(x) - 0.5 .* (1./x(:,2)).^2 .*(measurements(idx) - model(x(:,1),displacement(idx)))' *...
                                         (measurements(idx) - model(x(:,1),displacement(idx))) -...
                                          log(sqrt(2*pi).*x(:,2)); 
end

%% Perform Online Sequential Bayesian Updating:

% Initialise:
Nsamples = 1000;
step_size = 2;

% Start SEMC sampler:
tic;
SEMC = SEMCsampler('nsamples',Nsamples,'loglikelihoods',logL,...
                   'priorpdf',prior_pdf,'priorrnd',prior_rnd,...
                   'burnin',0,'stepsize',step_size);
semc_allsamples = SEMC.allsamples;
timeSEMC = toc;
fprintf('Time elapsed is for the SEMC sampler: %f \n',timeSEMC)

%% Analysis of the model updating:

least_squares = abs(displacement\measurements);
posterior_mean = zeros(time_step,size(semc_allsamples,2));
posterior_std = zeros(time_step,size(semc_allsamples,2));
posterior_bounds_k = zeros(time_step,2);
posterior_bounds_sigma = zeros(time_step,2);

for idx = 1:time_step
posterior_mean(idx,1) = mean(semc_allsamples(:,1,idx+1));
posterior_mean(idx,2) = mean(semc_allsamples(:,2,idx+1));

posterior_std(idx,1) = std(semc_allsamples(:,1,idx+1));
posterior_std(idx,2) = std(semc_allsamples(:,2,idx+1));

posterior_bounds_k(idx,:) = prctile(semc_allsamples(:,1,idx+1), [5, 95]);
posterior_bounds_sigma(idx,:) = prctile(semc_allsamples(:,2,idx+1), [5, 95]);
end
posterior_cov = (posterior_std./posterior_mean).*100;

% To plot the estimates of k and sigma wih the theoretical values:
figure;
subplot(1,2,1)
hold on; grid on; box on;
plot([0 size(posterior_mean,1)+1], [stiffness stiffness], 'k--', 'linewidth', 1)
plot([0 size(posterior_mean,1)+1], [least_squares least_squares], 'r--', 'linewidth', 1)
y_neg_a1 = abs(posterior_mean(:,1) - posterior_bounds_k(:,1)); % error in the negative y-direction
y_pos_a1 = abs(posterior_mean(:,1) - posterior_bounds_k(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean,1))', posterior_mean(:,1), y_neg_a1, y_pos_a1, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
legend('True value', 'Least squares estimate', 'SEMC estimated values', 'linewidth', 2)
xlim([0 16])
xlabel('Iteration, j')
ylabel('Stiffness, k [N/m]')
set(gca, 'fontsize', 18)

subplot(1,2,2)
hold on; grid on; box on;
plot([0 size(posterior_mean,1)+1], [noise_sd noise_sd], 'k--', 'linewidth', 1)
y_neg_b = abs(posterior_mean(:,2) - posterior_bounds_sigma(:,1)); % error in the negative y-direction
y_pos_b = abs(posterior_mean(:,2) - posterior_bounds_sigma(:,2)); % error in the positive y-direction
errorbar((1:size(posterior_mean,1))', posterior_mean(:,2), y_neg_b, y_pos_b, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
legend('True value', 'SEMC estimated values', 'linewidth', 2)
xlim([0 16])
xlabel('Iteration, j')
ylabel('Noise standard deviation, \sigma [N]')
set(gca, 'fontsize', 18)

%% SEMC Statistics

dim = size(semc_allsamples,2); % dimensionality of the problem
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
xlim([1 16])
ylim([0.1 0.8])
set(gca, 'fontsize', 18)

%% Save the data:

save('example_LinearStatic_SEMC');