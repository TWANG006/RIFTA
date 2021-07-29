function z_dct = dcti2fc(sx, sy, x, y)
%DCTI2FC two-dimentional Discrete Cosine Transform Integration 
%         (FFT + Complex implementation)
% INPUT: 
%		sx - slope in x-direction.
%		sy - slope in y-direction.
%		x - x coordinates.
%  		y - y coordinates.
% OUTPUT:
%		z_dct - height distribution after integration.
%
%   Copyright since 2014 by Lei Huang. All Rights Reserved.
%   E-mail: huanglei0114@gmail.com
%   2014-02-01 Original Version

% Regenerate sx and sy.
temp = [sx,fliplr(-sx)];
temp = [temp',flipud(temp)']';
sx = temp;

temp = [sy,fliplr(sy)];
temp = [temp',flipud(-temp)']';
sy = temp;

% Get the size of slope matrix.
[Ny,Nx] = size(sx);
nx = 1:Nx;
ny = 1:Ny;

% Get the unit distance in spatial domain.
% Note: If the dataset is not in rectangle, this algorithm fails.
dx = x(1,2)-x(1,1);
dy = y(2,1)-y(1,1);
% Get the total enlargement in spatial domain.
L0X = dx*Nx;
L0Y = dy*Ny;

% Calculate the coordinate in spectral domain.
u = ((nx-1)-Nx/2)/L0X;
v = ((ny-1)-Ny/2)/L0Y;
[fx,fy] = meshgrid(u,v); 
fx = fftshift(fx);
fy = fftshift(fy);

% Take the DST&DCT by using FFT.
U = fft2(sx+1j*sy);

% Calculate the coefficients.
C = 1./(1j*2*pi*(fx+1j*fy));
% Avoid Inf, which will fail IDCT.
C(1,1) = 0;

clear fx fy nx ny sx sy x y u v temp;

% Calculate C{z}.
Cz = U.*C;
clear U C;

% Take the 2D IDCT with IFFT.
z = real(ifft2(Cz));
z_dct = z(1:Ny/2,1:Nx/2);

end % End the function.



