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
% Here, Bayesian Model Updating is performed via TMCMC sampler and TEMCMC
% sampler as examples.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;

%% 1) Defining the variables:

Nmeasurements = 15;
displacement = unifrnd(0.02,0.08,Nmeasurements,1);        % displacement [m]
stiffness = 263;                                          % True value of k [N/m] 

%% 2) Defining the model:

model = @(k,d) -k.*d;

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
prior_sigma = @(x) unifpdf(x,1e-05,2); % Defining the Prior distribution for sigma.
prior_pdf = @(x) prior_k(x(:,1)).*prior_sigma(x(:,2));

% To draw random samples from the Prior:
prior_rnd = @(N) [unifrnd(1,1000,N,1), unifrnd(1e-05,2,N,1)]; 

%% Define the likelihood function cell array
% This is to indicate measurements obtained separately at different
% time-step.

time_step = 15;

% To create array of Likelihood functions:
logl = cell(time_step,1);
for i = 1:time_step
idx = i; % index variable
logl{i} = @(x) - 0.5 .* (1./x(:,2)).^2 .*(measurements(idx) - model(x(:,1),displacement(idx)))' *...
                                         (measurements(idx) - model(x(:,1),displacement(idx))) -...
                                          log(sqrt(2*pi).*x(:,2)); 
end

%% Perform Online Bayesian Updating:

% Initialise:
Nsamples = 1000;
priorPDF = @(x) prior_pdf(x);
TMCMC_priorsamps = prior_rnd(Nsamples);
TEMCMC_priorsamps = prior_rnd(Nsamples);

for t = 1:time_step

fprintf('Commence Online Updating Stage; Time-step: %2d \n',i)    
    
% Update Prior with new Likelihoods:
logL = logl{t};

% Run the TMCMC sampler:
tic;
TMCMC = TMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
               'priorpdf',priorPDF,'priorsamps',TEMCMC_priorsamps,'burnin',0);
timeTMCMC(t) = toc;
fprintf('Time elapsed is for the TMCMC sampler: %f \n',timeTMCMC(end))

% Run the TEMCMC sampler:
tic;
TEMCMC = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
               'priorpdf',priorPDF,'priorsamps',TEMCMC_priorsamps,'burnin',0);
timeTEMCMC(t) = toc;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC(end))

% Store posterior samples/statistics from loop:
Posterior_TMCMC(:,:,t) = TMCMC.samples; 
TMCMC_posterior_mean(:,:,t) = mean(TMCMC.samples);
TMCMC_posterior_stdev(:,:,t) = std(TMCMC.samples);
TMCMC_posterior_COV(:,:,t) = (std(TMCMC.samples)./mean(TMCMC.samples)).*100;

Posterior_TEMCMC(:,:,t) = TEMCMC.samples;
TEMCMC_posterior_mean(:,:,t) = mean(TEMCMC.samples);
TEMCMC_posterior_stdev(:,:,t) = std(TEMCMC.samples);
TEMCMC_posterior_COV(:,:,t) = (std(TEMCMC.samples)./mean(TEMCMC.samples)).*100;

% Prepare variables for the next loop:
priorPDF = @(x) priorPDF(x) .* exp(logL(x));
TMCMC_priorsamps = TMCMC.samples;
TEMCMC_priorsamps = TEMCMC.samples;

fprintf('End of Online Updating Stage; Time-step: %2d \n',t) 
end

%% Report the statistics of the final Posterior:

% TMCMC:
fprintf('Estimation of k via TMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TMCMC_posterior_mean(1,1,end), TMCMC_posterior_stdev(1,1,end), TMCMC_posterior_COV(1,1,end))
fprintf('Estimation of sigma via TMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TMCMC_posterior_mean(1,2,end), TMCMC_posterior_stdev(1,2,end), TMCMC_posterior_COV(1,2,end))
     
% TEMCMC:
fprintf('Estimation of k via TEMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TEMCMC_posterior_mean(1,1,end), TEMCMC_posterior_stdev(1,1,end), TEMCMC_posterior_COV(1,1,end))
fprintf('Estimation of sigma via TEMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TEMCMC_posterior_mean(1,2,end), TEMCMC_posterior_stdev(1,2,end), TEMCMC_posterior_COV(1,2,end))

%% Analyse Posterior samples from each stage:

% Analyse Posterior samples from TMCMC:
for t = 1:time_step
figure();
subplot(1,2,1)
box on;
histogram(Posterior_TMCMC(:,1,t),50)
title(sprintf('TMCMC Posterior for k at Time-step = %2d \n', t))
xlabel('k [N/m]')
ylabel('Count')
xlim([min(Posterior_TMCMC(:,1,1)), max(Posterior_TMCMC(:,1,1))])
ylim([0, 110])
set(gca, 'Fontsize', 16)

subplot(1,2,2);
box on;
histogram(Posterior_TMCMC(:,2,t),50)
title(sprintf('TMCMC Posterior for \\sigma at Time-step = %2d \n', t))
xlabel('\sigma [N]')
ylabel('Count')
xlim([min(Posterior_TMCMC(:,2,1)), max(Posterior_TMCMC(:,2,1))])
ylim([0, 110])
set(gca, 'Fontsize', 16)
end

% Analyse Posterior samples from TEMCMC:
for t = 1:time_step
figure();
sibplot(1,2,1)
box on;
histogram(Posterior_TEMCMC(:,1,t),50)
title(sprintf('TEMCMC Posterior for k at Time-step = %2d \n', t))
xlabel('k [N/m]')
ylabel('Count')
xlim([min(Posterior_TEMCMC(:,1,1)), max(Posterior_TEMCMC(:,1,1))])
ylim([0, 110])
set(gca, 'Fontsize', 16)

subplot(1,2,2)
box on;
histogram(Posterior_TEMCMC(:,2,t),50)
title(sprintf('TEMCMC Posterior for \\sigma at Time-step = %2d \n', t))
xlabel('\sigma [N]')
ylabel('Count')
xlim([min(Posterior_TEMCMC(:,2,1)), max(Posterior_TEMCMC(:,2,1))])
ylim([0, 110])
set(gca, 'Fontsize', 16)
end

%% Model Update:

% To obtain 5th and 95th percentile of TMCMC samples:
bounds_tmcmc = prctile(Posterior_TMCMC(:,1,end), [5, 95]);
% To obtain 5th and 95th percentile of TEMCMC samples:
bounds_temcmc = prctile(Posterior_TEMCMC(:,1,end), [5, 95]);

figure();
subplot(1,2,1) % Plot Model Update results for TMCMC
hold on; box on; grid on;
for i = 1:length(Posterior_TMCMC)
plot(disp, model(Posterior_TMCMC(i,1,end),disp),'color','#C0C0C0', 'LineWidth', 1)
end
legend('TMCMC samples','Linewidth', 2)
scatter(displacement, measurements, 13, 'r', 'filled', 'DisplayName', 'Noisy measurements')
plot(disp, model(stiffness,disp), 'k --', 'LineWidth', 1, 'DisplayName', 'True measurements')
plot(disp, model(bounds_tmcmc(1),disp), 'm', 'LineWidth', 1, 'DisplayName', '5^{th} percentile')
plot(disp, model(bounds_tmcmc(2),disp), 'm', 'LineWidth', 1, 'DisplayName', '95^{th} percentile')
plot(disp, model(mean_tmcmc_k,disp), 'c', 'LineWidth', 1, 'DisplayName', 'Mean')
xlim([0.02 0.08])
xlabel('Displacement, d [m]')
ylabel('Force, F [N]')
set(gca, 'Fontsize', 16)

subplot(1,2,2) % Plot Model Update results for TEMCMC
hold on; box on; grid on;
for i = 1:length(Posterior_TEMCMC)
plot(disp, model(Posterior_TEMCMC(i,1,end),disp),'color','#C0C0C0', 'LineWidth', 1)
end
legend('TEMCMC samples','Linewidth', 2)
scatter(displacement, measurements, 13, 'r', 'filled', 'DisplayName', 'Noisy measurements')
plot(disp, model(stiffness,disp), 'k --', 'LineWidth', 1, 'DisplayName', 'True measurements')
plot(disp, model(bounds_temcmc(1),disp), 'm', 'LineWidth', 1, 'DisplayName', '5^{th} percentile')
plot(disp, model(bounds_temcmc(2),disp), 'm', 'LineWidth', 1, 'DisplayName', '95^{th} percentile')
plot(disp, model(mean_temcmc_k,disp), 'c', 'LineWidth', 1, 'DisplayName', 'Mean')
xlim([0.02 0.08])
xlabel('Displacement, d [m]')
ylabel('Force, F [N]')
set(gca, 'Fontsize', 16)

%% Save the data:

save('example_LinearStatic');
