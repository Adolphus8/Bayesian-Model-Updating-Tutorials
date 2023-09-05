function [output] = DenHartogHarmonic(beta_v)
%% Function-handle for Analytical plots of Tramissibility and Phase Angles:
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
%% Define key parameters:

% Define the non-dimensional parameters:
r_v = [0.01:0.01:0.9 0.901:0.001:0.999 1.001:0.001:1.099 1.1:0.025:2.6];
beta_v = [0 beta_v];

dt = pi/15;       % Time-step
t_half = 0:dt:pi; % Time half-period
N_cyc =  30;      % To be selected for FFT performance

%% Obtain the boundary values of Displacement Transmissibility and Phase Angles:

for ir = 1:length(r_v) % For each Frequency ratio value
r = r_v(ir);
    
% Response and damping functions:
U = sin(pi/r)/(r*(1+cos(pi/r))); % See Eq. (7)
V = 1/(1-r^2);                   % See Eq. (8)
    
% Boundaries:
sr = (r*sin(t_half/r)+U*r^2*(cos(t_half)-cos(t_half/r)))./sin(t_half); % See Eq. (9)
S = max(sr(1:end-1));

beta_bound = sqrt(V^2/(U^2+(S/r^2)^2)); % See Eq. (6)
X_bound = sqrt(V^2 - beta_bound^2*U^2); % Displacement Tranmissibility bound (see Eq. (11))
ph_bound = angleCalc(- beta_bound*U/V, X_bound/V, 'rad'); % Compute the bounds for Phase Angles

x_bound_half = X_bound*cos(t_half) + beta_bound*U*sin(t_half) + ...
beta_bound*(1 - cos(t_half/r) - U*r*sin(t_half/r)); % See Eq. (15)
y_bound_half = cos(t_half + ph_bound);              % See Eq. (16)
    
t_period = [t_half t_half(2:end)+pi];
x_bound_period = [x_bound_half -x_bound_half(2:end)];
y_bound_period = [y_bound_half -y_bound_half(2:end)];
    
t = t_period;
for in = 1:N_cyc - 1
t = [t t_period(2:end)+2*in*pi];
end

x_bound = [x_bound_period repmat(x_bound_period(2:end),1,N_cyc-1)];
y_bound = [y_bound_period repmat(y_bound_period(2:end),1,N_cyc-1)];

%% Fast-Fourier Transform Step:
    
Nt = length(t);         % No. of time-steps
fs = 1/dt;              % Frequency of data-collection
df = 1/t(end);          % Frequency steps
omega = 2*pi*(0:df:fs); % Value of omegas = 2*pi*f

% Compute the bounds in the time and frequency domains:    
x_bound_fft = fft(x_bound)/Nt;
y_bound_fft = fft(y_bound)/Nt;
    
i_peak = find(round(omega,3)>=1,1);
% Note: round(omega,3) rounds the omega value to the nearest thousandth (10^-3).
X_bound_peak = abs(x_bound_fft(i_peak));
Y_bound_peak = abs(y_bound_fft(i_peak));

% Identify the Den-Hartog's bound for Transmissibility (see Eq. (29)):
Tr_bound(ir) = X_bound_peak/Y_bound_peak; 

% Identify the Den-Hartog's bound for Phase Angles (see Eq. (30)):
phase_bound(ir) = rad2deg(angle(y_bound_fft(i_peak)/x_bound_fft(i_peak)));
            
% Compute the Transmissibility and Phase Angle Response for different beta values:
for ib = 1:length(beta_v)
beta = beta_v(ib);

if beta <= beta_bound || beta == 0   % In the continuous motion condition
X = sqrt(V^2 - (beta^2 * U^2));      % See Eq. (11)
ph = angleCalc(-beta*U/V,X/V,'rad'); % See Eq. (12)
x_half = X*cos(t_half)+beta*U*sin(t_half)+beta*(1-cos(t_half/r)-U*r*sin(t_half/r)); % See Eq. (15)
y_half = cos(t_half+ph);             % See Eq. (16)
            
% Post-processing step:          
x_period = [x_half -x_half(2:end)];
y_period = [y_half -y_half(2:end)];
 
x = [x_period repmat(x_period(2:end),1,N_cyc-1)];
y = [y_period repmat(y_period(2:end),1,N_cyc-1)];

y_fft = fft(y)/Nt;
x_fft = fft(x)/Nt;
            
Y_peak = abs(y_fft(i_peak));
X_peak = abs(x_fft(i_peak));

% Transimissibility response for r = r_v and beta = beta_v (see Eq. (29)):
Tr(ib,ir) = X_peak/Y_peak; 

% Phase Angle response for r = r_v and beta = beta_v (see Eq. (30)):
phase(ib,ir) = rad2deg(angle(y_fft(i_peak)/x_fft(i_peak))); 

else % In the stick-slip domain    
Tr(ib,ir) = NaN;
phase(ib,ir) = NaN;
end

end
end

%% Consolidate the output:
output.frequency_ratios = r_v;
output.trans = Tr;
output.phase_angles = phase;
output.trans_bound = Tr_bound;
output.phase_bound = phase_bound;
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
% theta= angleCalc(sin(2*pi/3),cos(2*pi/3),'rad')
% theta= 2.0944  [rad]
% --------------Disi A Jun 25, 2013
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
    else % If S(i) = 0
        theta(i) = theta(i) + pi;
    end
end

theta(i) = theta(i) .* cons;
end
end


