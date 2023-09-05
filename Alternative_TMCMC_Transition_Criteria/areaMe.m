function [outputArg1] = areaMe(D1,D2)
%AREAME Computes the area between two ECDFs
%   It does not work with a single datum.
%   
% . 
% . by The Liverpool Git Pushers
if length(D1)>length(D2)
    d1 = D2(:);
    d2 = D1(:);
else
    d1 = D1(:);
    d2 = D2(:);
end
[Pxs,xs] = ecdf_Lpool(d1);            % Compute the ecdf of the data sets
[Pys,ys] = ecdf_Lpool(d2);            
Pys_eqx = Pxs;
Pys_pure = Pys(2:end-1); % this does not work with a single datum
Pall = sort([Pys_eqx;Pys_pure]);
ys_eq_all = zeros(length(Pall),1);
ys_eq_all(1)=ys(1);
ys_eq_all(end)=ys(end);
for k=2:length(Pall)-1
    ys_eq_all(k,1) = interpCDF_2(ys,Pys,Pall(k));
end
xs_eq_all = zeros(length(Pall),1);
xs_eq_all(1)=xs(1);
xs_eq_all(end)=xs(end);
for k=2:length(Pall)-1
    xs_eq_all(k,1) = interpCDF_2(xs,Pxs,Pall(k));
end
diff_all_s = abs(ys_eq_all-xs_eq_all);
diff_all_s = diff_all_s(2:end);
diff_all_p = diff(Pall);
area = diff_all_s' * diff_all_p;
outputArg1 = area;
end


function [outputArg1] = interpCDF_2(xd,yd,pvalue)
%INTERPCDF Summary of this function goes here
%   Detailed explanation goes here
%   
% . 
% . by The Liverpool Git Pushers

% [yd,xd]=ecdf_Lpool(data);
beforr = diff(pvalue <= yd) == 1; % && diff(0.5>pv) == -1;
beforrr = [0;beforr(:)];
if pvalue==0
    xvalue = xd(1);
else
    xvalue = xd(beforrr==1);
end
outputArg1 = xvalue;
end


function [ps,xs] = ecdf_Lpool(x)

    xs = sort(x);
    xs = [xs(1);xs(:)];
    ps = linspace(0,1,length(xs))';
    
end