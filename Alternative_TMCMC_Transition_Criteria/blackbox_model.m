function [output] = blackbox_model(theta, frequency_ratios, driving_force)
%% Function handle for the Black-box model:
%
% This Black-box model takes the input of the epistemic parameters (Coulomb
% Force and Natural Frequency), driving frequencies of the rotor, and
% driving force amplitude of the rotor.
%
% Reference literatures:
% [1] L. Marino and A. Cicirello (2020). Experimental investigation of a single-
% degree-of-freedom system with Coulomb friction. Nonlinear Dynamics, 99(3), 
% 1781-1799. doi: 10.1007/s11071-019-05443-2
%
% [2] L. Marino, A. Cicirello, and D. A. Hills (2019). Displacement transmissibility
% of a Coulomb friction oscillator subject to joined base-wall motion. 
% Nonlinear Dynamics 98, 2595–2612. https://doi.org/10.1007/s11071-019-04983-x
%
%--------------------------------------------------------------------------
% Authors:
% Luca Marino          - luca.marino@lincoln.ox.ac.uk
% Adolphus Lye         - adolphus.lye@liverpool.ac.uk
%--------------------------------------------------------------------------
%
% Inputs:
% theta:               N x 1 input matrix of the epistemic time-varying Coulomb Friction [N];
% frequency_ratios:    N_e x 1 input vector of the frequency ratios used for the experiment [rad/s];
% driving_force:       Scalar value of the maximum driving force of the rotor on the structure [N];
%
% Output:
% output.frequency_ratios: N_e x 1 output vector of the dimensionless frequency ratios;
% output.phase_angles:     N_e x 1 output vector of the phase angles [deg];
%
%-------------------------------------------------------------------------%
%% Checks:
assert(size(theta,1) == 1);
assert(size(driving_force,1) == 1);
assert(size(driving_force,1) == size(driving_force,2));

%% Define the key parameters:

% Define the dimensionless Force ratio:
beta_v = theta(:,1) ./ driving_force;

% Define the dimensionless Frequency ratios:
r_v = frequency_ratios;

dt = pi/15;       % Time-step
t_half = 0:dt:pi; % Time half-period
N_cyc =  30;      % To be selected for FFT performance

%% Obtain the boundary values of Displacement Transmissibility and Phase Angles:

% Initialise empty output vectors:
phase = zeros(length(r_v),1); % Vector of response Phase Angles

for ir = 1:length(r_v) % For each Frequency ratio value
r = r_v(ir);
    
% Response and damping functions:
U = sin(pi/r)/(r*(1+cos(pi/r))); % See Eq. (7) of [1]
V = 1/(1-r^2);                   % See Eq. (8) of [1]
    
% Boundaries:
sr = (r*sin(t_half/r)+U*r^2*(cos(t_half)-cos(t_half/r)))./sin(t_half); % See Eq. (9) of [1]
S = max(sr(1:end-1));
beta_bound = sqrt(V^2/(U^2 + (S/r^2)^2)); % See Eq. (6) of [1]

t_period = [t_half t_half(2:end)+pi];
t = t_period;
for in = 1:N_cyc - 1
t = [t t_period(2:end)+2*in*pi];
end

%% Fast-Fourier Transform Step:
    
Nt = length(t);         % No. of time-steps
fs = 1/dt;              % Frequency of data-collection
df = 1/t(end);          % Frequency steps
omega = 2*pi*(0:df:fs); % Value of omegas = 2*pi*f
      
% Compute the Transmissibility and Phase Angle Response for different beta values:
beta = beta_v;

if beta <= beta_bound || beta == 0     % In the continuous motion condition
X = sqrt(V^2 - (beta^2 * U^2));        % See Eq. (11) of [1]
ph = angleCalc(-beta*U/V, X/V, 'rad'); % See Eq. (12) of [1]

% The half-period response of top plate (see Eq. (5) of [2]):
x_half = X*cos(t_half)+beta*U*sin(t_half)+beta*(1-cos(t_half/r)-U*r*sin(t_half/r)); 
% The half-period excitation of base plate (see Eq. (5) of [2]):
y_half = cos(t_half + ph);             
            
% Post-processing step:          
x_period = [x_half -x_half(2:end)];
y_period = [y_half -y_half(2:end)];

% Repeat the signal over N_cyc times:
x = [x_period repmat(x_period(2:end),1,N_cyc-1)];
y = [y_period repmat(y_period(2:end),1,N_cyc-1)];

y_fft = fft(y)/Nt;
x_fft = fft(x)/Nt;

% Phase Angle response for r = r_v and beta = beta_v (see Eq. (30) of [1]):
i_peak = find(round(omega,3)>=1,1);
% Note: round(omega,3) rounds the omega value to the nearest thousandth (10^-3).
phase(ir) = rad2deg(angle(y_fft(i_peak)/x_fft(i_peak))); 

else % In the stick-slip domain  
phase(ir) = NaN;
end

end

%% Consolidate the model outputs as a structure:
output.frequency_ratios = r_v;
output.phase_angles = phase;

end

function theta = angleCalc(S,C,out_mode)
%% This function computes the angle from sin and cos values (-180,180] degree.
%
% Usage: 
% theta = angleCalc(S,C,out_mode)
%
% Input:
% S:        Sine value of the angle
% C:        Cosine value of the angle
% out_mode: 'deg' OR 'rad'
% Note: default output mode is in degree 
%
% Output:
% theta: Angles in degrees or radians.
%
% Example:  
% theta = angleCalc(sin(-2*pi/3),cos(-2*pi/3))
% theta = -120;
% theta = angleCalc(sin(2*pi/3),cos(2*pi/3),'rad')
% theta = 2.0944  [rad]
%--------------------------Disi A Jun 25, 2013----------------------------%
%% Define the function:
if nargin < 3
    out_mode='deg';
end

if strcmp(out_mode,'deg')
    cons = 180/pi;
else
    cons = 1;
end

for i = 1:length(S)
theta(i) = asin(S(i));
if C(i) < 0
    if S(i) > 0
        theta(i) = pi - theta(i);
    elseif S(i) < 0
        theta(i) = - pi - theta(i);
    else % if S(i) = 0
        theta(i) = theta(i) + pi;
    end
end

theta(i) = theta(i) .* cons;
end
end


