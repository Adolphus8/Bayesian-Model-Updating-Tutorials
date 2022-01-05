function [output] = SMCsampler(varargin)
%% Sequential Monte Carlo Dynamical (SMC) sampler
%
% This program implements the original Sequential Monte Carlo (SMC) sampling
% class (see paper by N. Chopin (2002): A sequential particle filter method
% for static models - https://www.jstor.org/stable/3879283) and employs
% the use of the Affine-invariant Ensemble Sampler (AIES) proposed by 
% Goodman and Weare (2010) to update the samples at each iteration.
%
% This sampler function can be employed in Sequential Bayesian Model
% Updating problems involving:
% - Estimating time-invariant parameter(s) via Online Bayesian Model Updating;
% - Estimating time-varying parameter(s), following a recursive dynamic model;
% - Predicting the time-varying parameter(s) for the next time-step given
% data/observations up to the previous time-step.
%
%--------------------------------------------------------------------------
% Author:
% Adolphus Lye         - adolphus.lye@liverpool.ac.uk
%--------------------------------------------------------------------------

% Parse the information in the name/value pairs: 
pnames = {'nsamples','loglikelihoods','dynamic_model',...
          'priorpdf','priorrnd','burnin','lastburnin','thinchain'};

% Define default values:      
dflts =  {[], [], @(x) x, [], [], 0, 0, 3}; 

[nsamples,loglikelihoods,dynamic_model,priorpdf,prior_rnd,...
 burnin,lastBurnin,thinchain] = internal.stats.parseArgs(pnames, dflts, varargin{:});
   
%--------------------------------------------------------------------------
%
% Inputs:
% nsamples:       Scalar value of the number of samples to be generated from the Posterior;
% loglikelihoods: A M x 1 cell vector of likelihood functions containing the measurements at M different time-steps;
% dynamic_model:  A function-handle that relates theta(t+1) and theta(t), where t is the time-step. Output is N x dim; 
% priorpdf:       Function-handle of the Prior PDF;
% prior_rnd:      Function-handle of the Prior random number generator;
% burnin:         Number of burn-in for all iterations up to M-1;
% lastBurnin:     Number of burn-in for the last iteration;
% stepsize:       The stepsize for the Ensemble sampler in the updating step (this is the tuning parameter);
% thinchain:      Thin all the chains of the Ensemble sampler by only storing every k'th step (default=3);
% 
% Outputs:
% output.samples:    A N x dim matrix of Posterior samples;
% output.allsamples: A N x dim x (M+1) array of samples from all iterations;
% output.acceptance: A M x 1 vector of acceptance rates for all iterations;
% output.log_evidence: A (M+1) x 1 vector of the logarithmic of the evidence;
% output.step:       A M x 1 vector of step-size;
% output.indicator:  A M x 1 vector of indicators denoting if resampling
% has occured for any iterations (1 = Yes, 0 = No);
%
%--------------------------------------------------------------------------

%% Number of cores
if ~isempty(gcp('nocreate'))
    pool = gcp;
    Ncores = pool.NumWorkers;
    fprintf('SMC is running on %d cores.\n', Ncores);
end

%% Initialize: Obtain N samples from the Prior PDF

fprintf('Start SMC procedure ... \n');

prior_initial = priorpdf;     % Define initial Prior PDF
thetaj = prior_rnd(nsamples); % theta0 = N x dim
Dimensions = size(thetaj, 2); % Dimensionality of theta, dim

% Initialization of matrices and vectors:
thetaj1  = zeros(nsamples, Dimensions);
log_evidence = zeros(size(loglikelihoods,1)+1,1); % Initiate empty vector for log evidence
log_evidence(1) = 0;

acceptance = zeros(size(loglikelihoods,1),1);

% Samples from filter distribution, P(theta(t)|Data(1:t)):
allsamples = zeros(size(thetaj,1), size(thetaj,2), size(loglikelihoods,1)+1);
allsamples(:,:,1) = thetaj;

% Statistics from predictive distribution, P(theta(t+1)|Data(1:t)):
predictive_samples = zeros(size(thetaj,1), size(thetaj,2), size(loglikelihoods,1)); 

% Resampling indicator vector:
indicator = zeros(length(loglikelihoods), 1);
% Note: This indicator vector returns a 1 for the iteration(s) where
% resampling is initiated and 0 otherwise.

%% Main sampling loop
for iter = 1:length(loglikelihoods)

fprintf('SMC: Iteration j = %2d \n', iter);

loglikelihood = loglikelihoods{iter};

% Compute loglikelihood values for each sample:
logL = zeros(nsamples,1);
for l = 1:nsamples
logL(l) = loglikelihood(thetaj(l,:));
end

% Error check:
if any(isinf(logL))
error('The prior distribution is too far from the true region');
end

%% Compute weights of the samples, wj:

% To compute the nominal weights:
fprintf('Computing the weights ...\n');
wj = exp(logL);      

% To compute the log evidence for the current iteration:
log_evidence(iter+1) = log(mean(wj)) + log_evidence(iter);

% Check step for wj:
for i = 1:nsamples
if wj(i) == 0
wj(i) = 1e-100;
end
end

wj_norm = wj./sum(wj); % To normalise the weights

%% Check step - Compute the sum of wj_norm and see if it is < nsamples/2:

fprintf('Computing effective sample size ... \n');
Neff = 1/(sum(wj_norm.^2));
threshold = nsamples/2;

%% Resampling step (conditional if Neff < threshold):

if Neff < threshold
fprintf('Resampling step initiated ... \n');    

dx = randsample(nsamples, nsamples, true, wj_norm);

thetaj_resampled = zeros(nsamples, Dimensions);
for d = 1:nsamples
thetaj_resampled(d,:) = thetaj(dx(d),:); 
end

thetaj = thetaj_resampled;
wj_norm = (1/nsamples).*ones(nsamples,1);
indicator(iter) = 1;

end

%% Update the samples according to the current Posterior using MH sampler:

% Define the logposterior:
log_posterior = @(x) log(priorpdf(x)) + loglikelihood(x); 
    
% Weighted mean for Proposal distribution
mu = zeros(1, Dimensions);
for l = 1:nsamples
mu = mu + wj_norm(l)*thetaj(l,:); % 1 x N
end

% Covariance matrix for Proposal distribution:
cov_gauss = zeros(Dimensions);
for k = 1:nsamples
tk_mu = thetaj(k,:) - mu;
cov_gauss = cov_gauss + wj_norm(k)*(tk_mu'*tk_mu);
end
assert(~isinf(cond(cov_gauss)),'Something is wrong with the likelihood.')

% Define the Proposal distribution:
proppdf = @(x,y) prop_pdf(x, y, cov_gauss, prior_initial); % q(x,y) = q(x|y).
proprnd = @(x)   prop_rnd(x, cov_gauss, prior_initial);

if iter == length(loglikelihoods)
burnin = lastBurnin;
end

%% Start N different Markov chains
fprintf('Markov chains ...\n\n');

idx = randsample(nsamples, nsamples, true, wj_norm);
for i = 1:nsamples      % For parallel, type: parfor
[thetaj1(i,:), acceptance_rate(i)] = mhsample(thetaj(idx(i),:), 1, ...
                                          'logpdf',  log_posterior, ...
                                          'proppdf', proppdf, ...
                                          'proprnd', proprnd, ...
                                          'thin',  thinchain, ...
                                          'burnin',  burnin);
end
fprintf('\n');
acceptance(iter) = mean(acceptance_rate); % To store the acceptance rate values

%% Prediction step: 

% Define the Predictive distribution of the samples, P(theta(t+1)|Data(t)):
predictive_samples(:,:,iter) = dynamic_model(thetaj1);

% Compute the Bandwidth vector for the kernel density function:
pred_samps = predictive_samples(:,:,iter);

bw = zeros(Dimensions,1); 
for dim = 1:Dimensions
bw(dim) = std(pred_samps(:,dim)) .* (4/((Dimensions + 2) .* nsamples)).^(1/(Dimensions + 4));
end

% Define the Predictive PDF, P(theta(t+1)|Data(t)) using mvksdensity:
pred_pdf = @(x) mvksdensity(pred_samps, x, 'Bandwidth', bw);

%% Prepare for the next iteration:

allsamples(:,:,iter+1) = thetaj1;
thetaj = pred_samps;
priorpdf = @(x) pred_pdf(x);

end

%% Description of outputs:

output.samples = thetaj;                % To only show samples from the final filter distribution
output.allsamples = allsamples;         % To only show all filter samples across all iterations
output.prediction = predictive_samples; % To only show all prediction samples across all iterations
output.acceptance = acceptance;         % To show the mean acceptance rates for all iterations
output.log_evidence = log_evidence;     % To show the (M+1) x 1 vector of the logarithmic of the evidence;
output.indicator = indicator;           % To indicate the iterations whereby resampling took place (denoted by 1s).

fprintf('End of SMC procedure. \n\n');

return; % End

function proppdf = prop_pdf(x, mu, covmat, box)
% This is the Proposal PDF for the Markov Chain.

% Box function is the Prior PDF in the feasible region. 
% So if a point is out of bounds, this function will
% return 0.

proppdf = mvnpdf(x, mu, covmat).*box(x); %q(x,y) = q(x|y).

return;


function proprnd = prop_rnd(mu, covmat, box)
% Sampling from the proposal PDF for the Markov Chain.

while true
proprnd = mvnrnd(mu, covmat, 1);
if box(proprnd)
% The box function is the Prior PDF in the feasible region.
% If a point is out of bounds, this function will return 0 = false.
break;
end
end

return