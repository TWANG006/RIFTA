function [Zres,Zf,f] = remove_surface(X, Y, Z)
% Function:
%   [Z,Zf,f] = remove_surface(X,Y,Z)
%
% Purpose:
%   This function removes the tip and tilt from the input surface error map
%   Z and returns the coefficients
%
% Inputs:
%   X: X meshgrid coordinates [m]
%   Y: Y meshgrid coordinates [m]
%   Z: Height error map [m]
%
% Outputs:
%   Z: Surface error map after removing tip and tilt
%   Zf: Fitted surface error map 
%   f: the fitting coefficients
%
% Info:
%   Contact: huanglei0114@gmail.com (Dr. HUANG Lei)
%   Copyright reserved.

% valid data only
idx = isfinite( Z(:) );
z = Z(idx);
x = X(idx);
y = Y(idx);

% least squares method
H = [ones(size(x)), ...
    x, y, ...
    ];

% f = pinv(H)*z;
f = double(H)\double(z);

% fitting
Zf = f(1) ...
    + f(2)*X + f(3)*Y ...
    ;

% residual
Zres = Z - Zf; 

end
