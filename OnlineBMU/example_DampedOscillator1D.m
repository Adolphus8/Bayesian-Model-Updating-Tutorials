%% The SEMC sampler
%
% The SEMC sampler is based on the original Sequential Monte
% Carlo (SMC) sampling class (see paper by Moral et. al (2006): Sequential 
% Monte Carlo Samplers - https://www.jstor.org/stable/3879283) and employs
% the use of the Affine-invariant Ensemble Sampler (AIES) proposed by 
% Goodman and Weare (2010) to update the samples at each iteration.
%
%% Bayesian Inference of a Time-varying parameter:
%
% In this example, we will evaluate the performance of the SEMC sampler
% in estimating and predicting a time-varying parameter at different time-step. 
% The system of interest will be a simple damped spring-mass system. The 
% dynamical data of the displacement of the mass from its rest position will 
% be obtained between t=0s and t=5s at intervals of dt=0.01s. The system is
% taken to be underdamped.
%
% The simple displacement model is:
%
% x(t) = A*exp(-gamma*t)*cos(omega*t + d); we set d = 0 rad for simplicity;
%
% omega = sqrt((k./m) - (c./(2.*m)).^2); 
% gamma = c./(2.*m); 
% where m = 0.3 kg (mass of the block attached to spring);
%
% The spring stiffness is time-varying and weakens over time such that:
%
% k(t) = 1.5*(1 + exp(-alpha*t)); where alpha = 0.03 s^-1 (rate parameter);
%
% Discretizing the above equation, we get the following discretized
% dynamic equation:
%
% k(t+1) = exp(-alpha*t).*(k(t) - 1.5) + 1.5;
%
% The spring stiffnes is checked every month and it is assumed 
% that during the process of data-collection at every "inspection" time-step, 
% t_i, the value of k is constant.
%
%% Define the parameters and random variables:

m = 0.3;           % Mass of the blocks in [kg]
c = 0.55;          % Damping coefficient of the spring-mass system [Ns/m]
A = 0.05;          % Displacement of the spring-mass system [m]
t_d = (0:0.01:5)'; % Time-input of the dynamical data[s]
t_i = (0:1:12)';   % Time-input of the inspection of the spring [mth]

%% Define the models:

% Define model for the spring stiffness, k:
alpha = 0.03;    % Rate parameter [s^-1]
k = @(t) 1.5.*(1 + exp(- alpha .* t));

% Define model for the eigenfrequency, omega:
omega = @(t,m,c) sqrt((k(t)./m) - (c./(2.*m)).^2);

% Define model for simple harmonic oscillator:
delta = 0; % Initial phase term when time starts [rad] 
displacement = @(t_d,t_i) A.*exp(- (c./(2.*m)).*t_d).*cos((omega(t_i,m,c) .* t_d) + delta); % in [m]

%% Generate noisy measurements of dynamical Displacement:
sigma = 0.005; % The noise in the measurement is set at 10% of the amplitude [m]

% To generate noisy data-set of the dynamical system for different time
% steps, t_i:
data = zeros(size(t_d,1),size(t_i,1)); 
for idx = 1:length(t_i)
for jdx = 1:length(t_d)
data(jdx,idx) = displacement(t_d(jdx),t_i(idx)) + sigma .* randn(1,1);
end
end

% To compute the root-mean-square-error of the data relative to the
% theoretial model of the SHM at each time-step t_i:
rmse = zeros(length(t_i),1);
for idx = 1:length(t_i)
square_error = (data(:,idx) - displacement(t_d(1:length(t_d)),t_i(idx))).^2;
rmse(idx) = sqrt(mean(square_error));
end

% To plot the 2D graphs of the dynamical data:
figure;
hold on; grid on; box on;
for idx = 1:length(t_i)
plot(t_d, data(:,idx), 'linewidth', 1)
end
xlabel('Time, t_d [s]')
ylabel('Displacement, x(t_d) [m]')
xlim([0 5])
ylim([-0.1 0.1])
legend('t_i = 0 mth', 't_i = 1 mth', 't_i = 2 mth', 't_i = 3 mth', 't_i = 4 mth',...
       't_i = 5 mth', 't_i = 6 mth', 't_i = 7 mth', 't_i = 8 mth', 't_i = 9 mth',...
       't_i = 10 mth', 't_i = 11 mth', 't_i = 12 mth', 'LineWidth', 2)
set(gca, 'fontsize', 15)
hold off

% To plot the theoretical model of k(t) and c(t):
figure;
hold on; grid on; box on;
plot(t_i, k(t_i), 'k--', 'linewidth', 1)
xlabel('Inspection time, t_i [mth]')
ylabel('Stiffness, k(t_i) [N/m]')
xlim([0 12])
set(gca, 'fontsize', 20)

%% Bayesian Model Updating set-up:

% Define the prior distribution:
lowerBound = 0.001; 
upperBound = 100;

prior_pdf = @(x) unifpdf(x,lowerBound,upperBound); % Prior PDf for k

prior_rnd = @(N) unifrnd(lowerBound,upperBound,N,1);

% Define the likelihood function:
scale = 10;
measurement_model = @(x,t_d) A.*exp(- (c./(2.*m)).*t_d).*...
                            cos((sqrt((x./m) - (c./(2.*m)).^2).* t_d) + delta); % in [m]
             
loglike = @(x,t_i) - 0.5 .* (1./(scale.*rmse(t_i))).^2 .*(sum((data(:,t_i) - measurement_model(x, t_d)).^2));

logL = cell(length(t_i),1);
for idx = 1:length(t_i)
logL{idx} = @(x) loglike(x, idx);
end
                                                 
% Define the dynamical model for k(t+1)|k(t):
dynamic_model = @(k_old) exp(- alpha).*(k_old - 1.5) + 1.5;

% Define the inverse model for k(t)|k(t+1):
inverse_model = @(k_new) exp(alpha).*(k_new - 1.5) + 1.5;

%% Perform Online Sequential Bayesian Updating:

% Initialise:
Nsamples = 1000;

% Start SEMC sampler:
tic;
SEMC = SEMCsampler('nsamples',Nsamples,'loglikelihoods',logL,...
                   'dynamic_model',dynamic_model,'inverse_model',inverse_model,...
                   'priorpdf',prior_pdf,'priorrnd',prior_rnd,'burnin',0);
semc_allsamples = SEMC.allsamples;
semc_prediction = SEMC.prediction;
timeSEMC = toc;
fprintf('Time elapsed is for the SEMC sampler: %f \n',timeSEMC)

%% Analysis of the model updating:

posterior_mean = zeros(length(t_i),size(semc_allsamples,2));
posterior_std = zeros(length(t_i),size(semc_allsamples,2));
posterior_bounds_k = zeros(length(t_i),2);

prediction_mean = zeros(length(t_i),size(semc_allsamples,2));
prediction_std = zeros(length(t_i),size(semc_allsamples,2));
prediction_bounds_k = zeros(length(t_i),2);

for idx = 1:length(t_i)
posterior_mean(idx,1) = mean(semc_allsamples(:,1,idx+1));

prediction_mean(idx,1) = mean(semc_prediction(:,1,idx));

posterior_std(idx,1) = std(semc_allsamples(:,1,idx+1));

prediction_std(idx,1) = std(semc_prediction(:,1,idx));

posterior_bounds_k(idx,:) = prctile(semc_allsamples(:,1,idx+1), [5, 95]);

prediction_bounds_k(idx,:) = prctile(semc_prediction(:,1,idx), [5, 95]);
end
posterior_cov = (posterior_std./posterior_mean).*100;
prediction_cov = (prediction_std./prediction_mean).*100;

% Plotting the histogram for different time-steps:
figure;
hold on; box on;
for idx = 1:length(t_i)
histogram(semc_allsamples(:,1,idx+1),10)
end
xlim([2.5 3.15])
xlabel('Stiffness, k(t_i) [N/m]')
ylabel('Counts')
legend('t_i = 0 mth', 't_i = 1 mth', 't_i = 2 mth', 't_i = 3 mth', 't_i = 4 mth',...
       't_i = 5 mth', 't_i = 6 mth', 't_i = 7 mth', 't_i = 8 mth', 't_i = 9 mth',...
       't_i = 10 mth', 't_i = 11 mth', 't_i = 12 mth', 'LineWidth', 2)
set(gca, 'fontsize', 18)
hold off

% To plot the estimates of k(t_i) wih the theoretical model:
figure;
hold on; grid on; box on;
plot(t_i, k(t_i), 'k--', 'linewidth', 1)
y_neg_a1 = abs(posterior_mean(:,1) - posterior_bounds_k(:,1)); % error in the negative y-direction
y_pos_a1 = abs(posterior_mean(:,1) - posterior_bounds_k(:,2)); % error in the positive y-direction
errorbar(t_i, posterior_mean(:,1), y_neg_a1, y_pos_a1, '-s','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'linewidth',1);
y_neg_a2 = abs(prediction_mean(:,1) - prediction_bounds_k(:,1)); % error in the negative y-direction
y_pos_a2 = abs(prediction_mean(:,1) - prediction_bounds_k(:,2)); % error in the positive y-direction
errorbar((t_i(2:end)), prediction_mean(1:end-1,1), y_neg_a2(1:end-1), y_pos_a2(1:end-1), '-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red', 'linewidth',1);
legend('True value', 'SEMC estimated values', 'Predicted values', 'linewidth', 2)
xlim([0 12])
xlabel('Inspection time, t_i [mth]')
ylabel('Stiffness, k(t_i) [N/m]')
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
xlim([1 5])
set(gca, 'fontsize', 18)

%% Save the data:

save('example_DampedOscillator1D');
