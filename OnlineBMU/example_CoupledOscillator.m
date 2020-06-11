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
% Here, Bayesian Model Updating is performed via TMCMC sampler and TEMCMC
% sampler as examples.
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
xlim([0.9 1.4])
legend('Noisy eigenfrequencies', 'True eigenfrequency','LineWidth',2)
set(gca, 'fontsize', 15)
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

logL = @(x, mea) - 0.5 .* (1./x(:,3)).^2 .*(mea(:,1) - model_1([m,x(:,1)]))' *...
                                      (mea(:,1) - model_1([m,x(:,1)])) -...
                                       length(mea).*log(sqrt(2*pi).*x(:,3)) +...
            - 0.5 .* (1./x(:,4)).^2 .*(mea(:,2) - model_2([m, x(:,1), x(:,2)]))' *...
                                      (mea(:,2) - model_2([m, x(:,1), x(:,2)])) -...
                                       length(mea).*log(sqrt(2*pi).*x(:,4));

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

%% Perform Online Bayesian Updating:

% Initialise:
Nsamples = 1000;
priorPDF = @(x) prior_pdf(x);
TMCMC_priorsamps = prior_rnd(Nsamples);
TEMCMC_priorsamps = prior_rnd(Nsamples);

for t = 1:length(logl)

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

%% Plot the combined Scatterplot matrix:

for t = 1:time_step
figure();
subplot(1,2,1)
[~,ax1] = plotmatrix(Posterior_TMCMC(:,:,t));
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
title(sprintf('TMCMC Posterior at Time-step = %2d \n', t))
set(gca,'FontSize',16)

subplot(1,2,2)
[~,ax2] = plotmatrix(Posterior_TEMCMC(:,:,t));
for i=1:4
    ax2(i,1).FontSize = 16; 
    ax2(4,i).FontSize = 16; 
end
ax2(1,1).YLim = [-0.3, 2.7]; ax2(2,1).YLim = [-0.3, 2.3]; 
ax2(3,1).YLim = [-0.15, 1.15]; ax2(4,1).YLim = [-0.15,1.15]; 
ax2(4,1).XLim = [-0.3, 2.7]; ax2(4,2).XLim = [-0.3, 2.3]; 
ax2(4,3).XLim = [-0.15, 1.15]; ax2(4,4).XLim = [-0.15,1.15]; 

ylabel(ax2(1,1),'k [N/m]'); ylabel(ax2(2,1),'k_{12} [N/m]');
ylabel(ax2(3,1),'\sigma_1 [Hz]'); ylabel(ax2(4,1),'\sigma_2 [Hz]');
xlabel(ax2(4,1),'k [N/m]'); xlabel(ax2(4,2),'k_{12} [N/m]');
xlabel(ax2(4,3),'\sigma_1 [Hz]'); xlabel(ax2(4,4),'\sigma_2 [Hz]');
title(sprintf('TEMCMC Posterior at Time-step = %2d \n', t))
set(gca,'FontSize',16)
end

%% Report the statistics of the final Posterior:

% TMCMC:
fprintf('Estimation of k via TMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TMCMC_posterior_mean(1,1,end), TMCMC_posterior_stdev(1,1,end), TMCMC_posterior_COV(1,1,end))
fprintf('Estimation of k12 via TMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TMCMC_posterior_mean(1,2,end), TMCMC_posterior_stdev(1,2,end), TMCMC_posterior_COV(1,2,end))
fprintf('Estimation of sigma_1 via TMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TMCMC_posterior_mean(1,3,end), TMCMC_posterior_stdev(1,3,end), TMCMC_posterior_COV(1,3,end))
fprintf('Estimation of sigma_2 via TMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TMCMC_posterior_mean(1,4,end), TMCMC_posterior_stdev(1,4,end), TMCMC_posterior_COV(1,4,end))

% TEMCMC:
fprintf('Estimation of k via TEMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TEMCMC_posterior_mean(1,1,end), TEMCMC_posterior_stdev(1,1,end), TEMCMC_posterior_COV(1,1,end))
fprintf('Estimation of k12 via TEMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TEMCMC_posterior_mean(1,2,end), TEMCMC_posterior_stdev(1,2,end), TEMCMC_posterior_COV(1,2,end))
fprintf('Estimation of sigma_1 via TEMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TEMCMC_posterior_mean(1,3,end), TEMCMC_posterior_stdev(1,3,end), TEMCMC_posterior_COV(1,3,end))
fprintf('Estimation of sigma_2 via TEMCMC: %4.2f; standard deviation: %4.2f; COV: %3.2f%% \n',...
TEMCMC_posterior_mean(1,4,end), TEMCMC_posterior_stdev(1,4,end), TEMCMC_posterior_COV(1,4,end))

%% Model Update

update_model_1 = @(x) sqrt(x./m);
update_model_2 = @(x) sqrt((x(:,1) + 2.*x(:,2))./m);

figure;
subplot(1,2,1) % Plot Model Update results for TMCMC
hold on; box on; grid on
scatter(update_model_1(Posterior_TMCMC(:,1,5)),...
update_model_2([Posterior_TMCMC(:,1,5),Posterior_TMCMC(:,2,5)]), 10, 'b', 'filled')
scatter(measurements(:,1), measurements(:,2), 10, 'r', 'filled');
plot(model_1([m,k]), model_2([m,k,k_12]), 'k +','LineWidth', 2);
xlabel('\omega_1^{noisy} [Hz]')
ylabel('\omega_2^{noisy} [Hz]')
legend('TMCMC Model Update','Noisy eigenfrequencies', 'True eigenfrequency','LineWidth',2)
set(gca, 'fontsize', 15)

subplot(1,2,2) % Plot Model Update results for TEMCMC
hold on; box on; grid on
scatter(update_model_1(Posterior_TEMCMC(:,1,5)),...
update_model_2([Posterior_TEMCMC(:,1,5),Posterior_TEMCMC(:,2,5)]), 10, 'b', 'filled')
scatter(measurements(:,1), measurements(:,2), 10, 'r', 'filled');
plot(model_1([m,k]), model_2([m,k,k_12]), 'k +','LineWidth', 2);
xlabel('\omega_1^{noisy} [Hz]')
ylabel('\omega_2^{noisy} [Hz]')
legend('TEMCMC Model Update','Noisy eigenfrequencies', 'True eigenfrequency','LineWidth',2)
set(gca, 'fontsize', 15)

%% Save the data:

save('example_CoupledOscillator');
