function [Zx_fitted, Zy_fitted] = tif_gaussian_slope_2d(...
    X, ... X coordiantes
    Y, ... Y coordinates
    t, ... dwell time
    params... [A, sigmax, sigmay, ux1, uy1, ux2, uy2, ...]
    )
% Function:
%   [Zx_fitted, Zy_fitted] = tif_gaussian_slope_2d(X, Y, t, params)
% Purpose:
%   Slopes of the 2D Gaussian Beam removal function model:
%       Zx(X, Y) = Z(x,y)*(x0-x)/sigmax^2
%       Zy(X, Y) = Z(x,y)*(y0-y)/sigmay^2
%
% Info:
%   Contact: tianyiwang666@gmail.com (Dr WANG Tianyi)
%   Copyright reserved.
%--------------------------------------------------------------------------
% Get the parameters
A = params(1);
sigmax = params(2);     sigmay = params(3);
ux = params(4:2:end);   uy = params(5:2:end);

% Feed the result
Zx_fitted = zeros(size(X));
Zy_fitted = zeros(size(Y));

for i = 1:length(t)
    Zx_fitted(:, :, i) = ...
        (ux(i)-X(:,:,i))./sigmax.^2.*(A*t(i)*exp(-((X(:, :, i)-ux(i)).^2/(2*sigmax.^2) + (Y(:, :, i)-uy(i)).^2/(2*sigmay.^2))));
    Zy_fitted(:, :, i) = ...
        (uy(i)-Y(:,:,i))./sigmay.^2.*(A*t(i)*exp(-((X(:, :, i)-ux(i)).^2/(2*sigmax.^2) + (Y(:, :, i)-uy(i)).^2/(2*sigmay.^2))));
end

end