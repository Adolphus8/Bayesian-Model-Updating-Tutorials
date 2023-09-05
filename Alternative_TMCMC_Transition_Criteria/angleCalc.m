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