function logL = loglikelihood(theta, blackbox_model, phase_angle_data, frequency_data, frequency_ratio_exp, frequency_ratio_nom)
%% Function-handle of the Loglikelihood function:  
% This function-handle computes the log-likelihood values.
%-------------------------------------------------------------------------%
%
% Inputs:
% theta:        N x 4 input matrix of the epistemic parameters whereby:
% - theta(:,1): First dimension is that of the time-varying Coulomb Friction [N];
% - theta(:,2): Second dimension is that of the static Natural Frequency [Hz];
% - theta(:,3): Third dimension is that of the static noise for Phase Angle measurements [deg];
% - theta(:,4): Fourth dimension is that of the static noise for Frequency ratio measurements;
% Note: N is the sample size, the number of theta to generate from the posterior.
%
% blackbox_model:       Blackbox function-handle (function of theta) used for model evaluation;
% phase_angle_data:     N_e x 1 input vector of the phase angles measured from the experiment [deg];
% frequency_data:       N_e x 1 input vector of the driving frequencies used for the experiment [rad/s];
% frequency_ratio_nom:  N_e x 1 input vector of nominal frequency ratio;
% frequency_ratio_exp:  N_e x 1 input vector of experimental frequency ratio;
% Note: N_e is the number of experimental data taken.
%
% Output:
% logL:                 N x 1 vector of loglikelihood output values;
%
%-------------------------------------------------------------------------%
%% Define the Function-handle:

% Initiate the empty vector of logL:
logL = zeros(size(theta,1),1);

for i = 1:size(theta,1)
%% Generate the model output:
model_output = blackbox_model(theta(i,1), frequency_ratio_exp);

% Generate N_e x 1 model output of the Phase angles:
phase_angle_model = model_output.phase_angles; 

%% Compute the loglikelihood for Phase Angles:

logL_phi = - 0.5 .* (1./(theta(i,3)).^2) .* sum((phase_angle_data - phase_angle_model).^2) - ...
         size(phase_angle_data,1).*log(sqrt(2*pi).*theta(i,3));
     
%% Compute the loglikelihood for Frequency Ratios:

% Generate N_e x 1 model output of the frequency ratios:
freq_ratio_model = frequency_data./theta(i,2); 

logL_r = - 0.5 .* (1./(theta(i,4)).^2) .* sum((frequency_ratio_nom - freq_ratio_model).^2) - ...
         size(frequency_data,1).*log(sqrt(2*pi).*theta(i,4));
     
%% Compute the overall loglikelihood value:

logL(i) = logL_phi + logL_r;

% Set logL(i) = -1e10 if logL is NaN or Inf:
if isnan(logL(i)) || isinf(logL(i))
logL(i) = -1e10;
end

end
end

