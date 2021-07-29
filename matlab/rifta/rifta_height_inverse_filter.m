function T = rifta_height_inverse_filter(...
    Zdg, ... height to remove on dwell grid
    B,... BRF [m/s]
    gamma...gamma for the thresholding
    )
% 2D inverse filtering for deconvoltuion based on surface height Zdg
%
% Output:
%   T: the dwell time map  [s]

[m, n] = size(Zdg);    % size of the height in dwell grid

% Perform FFT
FZ = fft2(Zdg);
FB = fft2(B, m, n);

% Thresholding
sFB = FB.*(abs(FB)>0) + 1/gamma*(abs(FB)==0);
iFB = 1./sFB;
iFB = iFB.*(abs(sFB)*gamma>1)+gamma*(abs(sFB)*gamma<=1);

% Inverse filtering
T = real(ifft2(iFB.*FZ));
T(T<0) = 0;

end