function [output] = SEMCsampler(varargin)
%% Sequential Ensemble Monte Carlo Dynamical (SEMC) sampler
%
% This program implements a method based on the original Sequential Monte
% Carlo (SMC) sampling class (see paper by Moral et. al (2006): Sequential 
% Monte Carlo Samplers - https://www.jstor.org/stable/3879283) and employs
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
pnames = {'nsamples','loglikelihoods','dynamic_model','inverse_model',...
          'priorpdf','priorrnd','burnin','lastburnin','stepsize','thinchain'};

% Define default values:      
dflts =  {[], [], @(x) x, @(x) x, [], [], [], 0, 2, 3}; 


[nsamples,loglikelihoods,dynamic_model,inverse_model,priorpdf,prior_rnd,...
 burnin,lastBurnin,stepsize,thinchain] = internal.stats.parseArgs(pnames, dflts, varargin{:});
   
%--------------------------------------------------------------------------
%
% Inputs:
% nsamples:       Scalar value of the number of samples to be generated from the Posterior;
% loglikelihoods: A M x 1 or 1 x M cell vector of likelihood functions containing the measurements at M different time-steps;
% dynamic_model:  A function-handle that relates theta(t+1) and theta(t), where t is the time-step. Output is N x dim; 
% inverse_model:  A function-handle that relates theta(t) and theta(t+1), where t is the time-step. Output is N x dim;
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
% output.step:       A M x 1 vector of step-size;;
%
%--------------------------------------------------------------------------

%% Initialize: Obtain N samples from the Prior PDF

fprintf('Start SEMC procedure ... \n');

count = 1; % Initiate counter
thetaj = prior_rnd(nsamples); % theta0 = N x dim
Dimensions = size(thetaj, 2); % Dimensionality of theta, dim

% Initialization of matrices and vectors:
thetaj1  = zeros(nsamples, Dimensions);

step = zeros(size(loglikelihoods,1)+1,1);
step(count) = stepsize;

acceptance = zeros(size(loglikelihoods,1),1);

% Samples from filter distribution, P(theta(t)|Data(1:t)):
allsamples = zeros(size(thetaj,1), size(thetaj,2), size(loglikelihoods,1)+1);
allsamples(:,:,1) = thetaj;

% Samples from predictive distribution, P(theta(t+1)|Data(1:t)):
predictive_samples = zeros(size(thetaj,1), size(thetaj,2), size(loglikelihoods,1)); 

% Pre-sampling error check:
test_func = @(in) dynamic_model(inverse_model(in));
in = (1:10)'.*ones(10,size(thetaj, Dimensions));
if test_func(in) ~= in
error('Please check dynamic_model and inverse_model again');
end

%% Main sampling loop
for iter = 1:length(loglikelihoods)

fprintf('SEMC: Iteration j = %2d \n', iter);

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

fprintf('Computing the weights ...\n');
wj = exp(logL);        % To compute the nominal weights

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

end

%% Update the samples according to the current Posterior using EMCMC sampler:

% Define the logposterior:
log_posterior = @(x) log(priorpdf(x)) + loglikelihood(x); 

if iter == length(loglikelihoods)
burnin = lastBurnin;
end

% Start nsamples different Markov chains:
fprintf('Markov chains ...\n\n');
idx = randsample(nsamples, nsamples, true, wj_norm);

% Define the starting sampels of each of the nsamples markov chains:
start = zeros(nsamples, Dimensions);
for i = 1:nsamples
start(i,:) = thetaj(idx(i), :);
end

% Initiate the EMCMC sampler:
[samples,logp,acceptance_rate] = EMCMCsampler(start, log_posterior, 1, priorpdf, ...
                                             'StepSize', stepsize,...
                                             'BurnIn', burnin,...
                                             'ThinChain', thinchain); 
samples_nominal = permute(samples, [2 1 3]);

% To compress thetaj1 into a nsamples x dim vector:
thetaj1 = samples_nominal(:,:)';

acceptance(count) = acceptance_rate; % To store the acceptance rate values

%% Prediction step: 

% Define the Predictive distribution of the samples, P(theta(t+1)|Data(t)):
predictive_samples(:,:,count) = dynamic_model(thetaj1);

% Define the Predictive PDF, P(theta(t+1)|Data(t)):
pred_pdf = @(x) exp(log_posterior(inverse_model(x)));

%% Prepare for the next iteration:

c_a = (acceptance_rate - ((0.21./Dimensions) + 0.23));
stepsize = stepsize.*exp(c_a);
    
count = count+1;
allsamples(:,:,count) = thetaj1;
step(count) = stepsize;
thetaj = thetaj1;
priorpdf = @(x) pred_pdf(x);

end

%% Description of outputs:

output.samples = thetaj;                % To only show samples from the final filter distribution
output.allsamples = allsamples;         % To only show all filter samples across all iterations
output.prediction = predictive_samples; % To only show all prediction samples across all iterations
output.acceptance = acceptance;         % To show the mean acceptance rates for all iterations
output.step = step;                     % To show the values of step-size

fprintf('End of SEMC procedure. \n\n');

return; % End

function [models,logP,acceptance]=EMCMCsampler(minit,logPfuns,Nsamples,box,varargin)
%% Cascaded affine invariant ensemble MCMC sampler. "The MCMC hammer"
%
% GWMCMC is an implementation of the Goodman and Weare 2010 Affine
% invariant ensemble Markov Chain Monte Carlo (MCMC) sampler. MCMC sampling
% enables bayesian inference. The problem with many traditional MCMC samplers
% is that they can have slow convergence for badly scaled problems, and that
% it is difficult to optimize the random walk for high-dimensional problems.
% This is where the GW-algorithm really excels as it is affine invariant. It
% can achieve much better convergence on badly scaled problems. It is much
% simpler to get to work straight out of the box, and for that reason it
% truly deserves to be called the MCMC hammer.
%
% (This code uses a cascaded variant of the Goodman and Weare algorithm).
%
% USAGE:
%  [models,logP]=gwmcmc(minit,logPfuns,mccount, Parameter,Value,Parameter,Value);
%
% INPUTS:
%     minit: an WxM matrix of initial values for each of the walkers in the
%            ensemble. (M:number of model params. W: number of walkers). W
%            should be atleast 2xM. (see e.g. mvnrnd).
%  logPfuns: a cell of function handles returning the log probality of a
%            proposed set of model parameters. Typically this cell will
%            contain two function handles: one to the logprior and another
%            to the loglikelihood. E.g. {@(m)logprior(m) @(m)loglike(m)}
%   mccount: What is the desired total number of monte carlo proposals per chain.
%            This is the total number per chain before burn-in.
%
% Named Parameter-Value pairs:
%   'StepSize': unit-less stepsize (default=2).
%   'ThinChain': Thin all the chains by only storing every N'th step (default=10)
%   'ProgressBar': Show a text progress bar (default=true)
%   'Parallel': Run in ensemble of walkers in parallel. (default=false)
%   'BurnIn': fraction of the chain that should be removed. (default=0)
%
% OUTPUTS:
%    models: A WxMxT matrix with the thinned markov chains (with T samples
%            per walker). T=~(mccount/p.ThinChain)*(1 - burnin_rate).
%    logP: A WxPxT matrix of log probabilities for each model in the
%            models. here P is the number of functions in logPfuns.
%
% Note on cascaded evaluation of log probabilities:
% The logPfuns-argument can be specifed as a cell-array to allow a cascaded
% evaluation of the probabilities. The computationally cheapest function should be
% placed first in the cell (this will typically the prior). This allows the
% routine to avoid calculating the likelihood, if the proposed model can be
% rejected based on the prior alone.
% logPfuns={logprior loglike} is faster but equivalent to
% logPfuns={@(m)logprior(m)+loglike(m)}
%
%
% References:
% Goodman & Weare (2010), Ensemble Samplers With Affine Invariance, Comm. App. Math. Comp. Sci., Vol. 5, No. 1, 65–80
% Foreman-Mackey, Hogg, Lang, Goodman (2013), emcee: The MCMC Hammer, arXiv:1202.3665
%
% WebPage: https://github.com/grinsted/gwmcmc
%
% -Aslak Grinsted 2015

persistent isoctave;  
if isempty(isoctave)
	isoctave = (exist ('OCTAVE_VERSION', 'builtin') > 0);
end

if nargin<3
    error('GWMCMC:toofewinputs','GWMCMC requires atleast 3 inputs.')
end
M=size(minit,2);
if size(minit,1)==1
    minit=bsxfun(@plus,minit,randn(M*5,M));
end


p=inputParser;
if isoctave
    p=p.addParamValue('StepSize',2,@isnumeric); %addParamValue is chosen for compatibility with octave. Still Untested.
    p=p.addParamValue('ThinChain',10,@isnumeric);
    p=p.addParamValue('ProgressBar',false,@islogical);
    p=p.addParamValue('Parallel',false,@islogical);
    p=p.addParamValue('BurnIn',0,@isnumeric);
    p=p.parse(varargin{:});
else
    p.addParameter('StepSize',2,@isnumeric); %addParamValue is chose for compatibility with octave. Still Untested.
    p.addParameter('ThinChain',10,@isnumeric);
    p.addParameter('ProgressBar',false,@islogical);
    p.addParameter('Parallel',false,@islogical);
    p.addParameter('BurnIn',0,@isnumeric);
    p.parse(varargin{:});
end
p=p.Results;

Nwalkers=size(minit,1);

if size(minit,2)*2>size(minit,1)
    warning('GWMCMC:minitdimensions','Check minit dimensions.\nIt is recommended that there be atleast twice as many walkers in the ensemble as there are model dimension.')
end

if p.ProgressBar
    progress=@textprogress;
else
    progress=@noaction;
end



Nkeep = Nsamples + p.BurnIn; % number of samples drawn per walker

models=nan(Nwalkers,M,Nkeep); % pre-allocate output matrix

models(:,:,1)=minit; % models: A WxMxT matrix, minit: A Mx(W*T) matrix

if ~iscell(logPfuns)
    logPfuns={logPfuns};
end

NPfun=numel(logPfuns);

%calculate logP state initial pos of walkers
logP=nan(Nwalkers,NPfun,Nkeep); %logP = WxPxT
for wix=1:Nwalkers
    for fix=1:NPfun
        v=logPfuns{fix}(minit(wix,:));
        if islogical(v) %reformulate function so that false=-inf for logical constraints.
            v=-1/v;logPfuns{fix}=@(m)-1/logPfuns{fix}(m); %experimental implementation of experimental feature
        end
        logP(wix,fix,1)=v;
    end
end

if ~all(all(isfinite(logP(:,:,1))))
    error('Starting points for all walkers must have finite logP')
end


reject=zeros(Nwalkers,1);

% models: A WxMxT matrix; logP: WxPxT matrix
curm = models(:,:,1);  %curm: W x M matrix
curlogP = logP(:,:,1); %curlogP: W x P matrix
progress(0,0,0)
totcount=Nwalkers;
for row = 1:Nkeep % number of samples drawn per walker
    for jj=1:p.ThinChain
        %generate proposals for all walkers
        rix = mod((1:Nwalkers)+floor(rand*(Nwalkers-1)),Nwalkers)+1; % pick a random partner (Nwalker x 1 vector)
        
        proposedm = zeros(Nwalkers, size(minit,2));  % Nwalkers x dim matrix
        zz = zeros(Nwalkers, 1);                     % Nwalkers x 1 vector
        for i = 1:Nwalkers
        while true
        zz(i) = ((p.StepSize - 1)*rand(1,1) + 1).^2/p.StepSize;  % scalar
        proposedm(i,:) = curm(rix(i),:) - bsxfun(@times,(curm(rix(i),:)-curm(i,:)),zz(i)); % Nwalkers x dim matrix
        if box(proposedm(i,:)) % The box function is the Prior PDF in the feasible region.
        % Note: If a point is out of bounds, this function will return 0 = false.
        break;
        end
        end
        end
        
        logrand=log(rand(Nwalkers,NPfun+1)); %moved outside because rand is slow inside parfor
        if p.Parallel
            %parallel/non-parallel code is currently mirrored in
            %order to enable experimentation with separate optimization
            %techniques for each branch. Parallel is not really great yet.
            %TODO: use SPMD instead of parfor.

            parfor wix=1:Nwalkers
                cp=curlogP(wix,:);
                lr=logrand(wix,:);
                acceptfullstep=true;
                proposedlogP=nan(1,NPfun);
                if lr(1)<(numel(proposedm(wix,:))-1)*log(zz(wix))
                    for fix=1:NPfun
                        proposedlogP(fix)=logPfuns{fix}(proposedm(wix,:)); 
                        if lr(fix+1)>proposedlogP(fix)-cp(fix) || ~isreal(proposedlogP(fix)) || isnan( proposedlogP(fix) )
                            acceptfullstep=false;
                            break
                        end
                    end
                else
                    acceptfullstep=false;
                end
                if acceptfullstep
                    curm(wix,:)=proposedm(wix,:); curlogP(wix,:)=proposedlogP;
                else
                    reject(wix)=reject(wix)+1;
                end
            end
        else %NON-PARALLEL
            for wix=1:Nwalkers
                acceptfullstep=true;
                proposedlogP=nan(1,NPfun);
                if logrand(wix,1)<(numel(proposedm(wix,:))-1)*log(zz(wix))
                    for fix=1:NPfun
                        proposedlogP(fix)=logPfuns{fix}(proposedm(wix,:));
                        if logrand(wix,fix+1)>proposedlogP(fix)-curlogP(wix,fix) || ~isreal(proposedlogP(fix)) || isnan(proposedlogP(fix))
                            acceptfullstep=false;
                            break
                        end
                    end
                else
                    acceptfullstep=false;
                end
                if acceptfullstep
                    curm(wix,:)=proposedm(wix,:); curlogP(wix,:)=proposedlogP;
                else
                    reject(wix)=reject(wix)+1;
                end
            end

        end
        totcount=totcount+Nwalkers;
        progress((row-1+jj/p.ThinChain)/Nkeep,curm,sum(reject)/totcount)
    end
    models(:,:,row)=curm;
    logP(:,:,row)=curlogP;
end
progress(1,0,0);

acceptance = 1 - (sum(reject)/totcount);

if p.BurnIn>0
    crop=p.BurnIn;
    models(:,:,1:crop)=[]; 
    logP(:,:,1:crop)=[];
end

function textprogress(pct,curm,rejectpct)
persistent lastNchar lasttime starttime
if isempty(lastNchar)||pct==0
    lasttime=cputime-10;starttime=cputime;lastNchar=0;
    pct=1e-16;
end
if pct==1
    fprintf('%s',repmat(char(8),1,lastNchar));lastNchar=0;
    return
end
if (cputime-lasttime>0.1)

    ETA=datestr((cputime-starttime)*(1-pct)/(pct*60*60*24),13);
    progressmsg=[183-uint8((1:40)<=(pct*40)).*(183-'*') ''];
    curmtxt=sprintf('% 9.3g\n',curm(1:min(end,20),1));
    progressmsg=sprintf('\nGWMCMC %5.1f%% [%s] %s\n%3.0f%% rejected\n%s\n',pct*100,progressmsg,ETA,rejectpct*100,curmtxt);

    fprintf('%s%s',repmat(char(8),1,lastNchar),progressmsg);
    drawnow;lasttime=cputime;
    lastNchar=length(progressmsg);
end

function noaction(varargin)