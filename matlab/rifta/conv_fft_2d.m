function R = conv_fft_2d(S, K)
% Purpose:
%	Implement 2D convolution using FFT whose results conincide with conv2 'same'
%
% Inputs:
%   S: signal
%   K: kernel
%
% Output:
%   R: convolved signal
%
% Info:
%   Contact: tianyiwang666@gmail.com (Dr WANG Tianyi)
%   Copyright reserved.

%% Compute the FFT padding size
% Signal size
[mS, nS] = size(S);

% Kernal size
[mK, nK] = size(K);

% Paddign size
m = mS + mK - 1;
n = nS + nK - 1;

%% Padding R and B
S = zero_pad(S, m, n);
K = zero_pad(K, m, n);

%% Perform FFT & convolution
R = real(ifft2(fft2(S).*fft2(K)));

%% Crop the correct portion to recover the same size of S
R = conv_fft_2d_crop(R, mS, nS, mK, nK);

end

function R_crop = conv_fft_2d_crop(R_crop, mS, nS, mK, nK)
%% Crop the correct portion to recover the same size of S

if mod(mK,2)==1 && mod(nK,2)==1
    hmK = (mK-1)/2;
    hnK = (nK-1)/2;
    R_crop = R_crop(1+hmK:mS+hmK,1+hnK:nS+hnK);
elseif mod(mK,2)==0 && mod(nK,2)==1
    hmK = mK/2;
    hnK = (nK-1)/2;
    R_crop = R_crop(1+hmK:mS+hmK,1+hnK:nS+hnK);
elseif mod(mK,2)==1 && mod(nK,2)==0
    hmK = (mK-1)/2;
    hnK = nK/2;
    R_crop = R_crop(1+hmK:mS+hmK,1+hnK:nS+hnK);
else
    hmK = mK/2;
    hnK = nK/2;
    R_crop = R_crop(1+hmK:mS+hmK,1+hnK:nS+hnK);  
end

end