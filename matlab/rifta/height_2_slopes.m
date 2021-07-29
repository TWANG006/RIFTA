function [Zx, Zy] = height_2_slopes(...
    X,... X coordinates
    Y,... Y coordinates [m]
    Z,... height map [m]
    mpp,... [m/pixel]
    spot_sz... spot size that used to calculate the window size [m]
    )
% Generate the slope maps from height map, mimicking the NSP measurement
% results

%% 0. Get the required sizes
[m, n] = size(Z);   % obtaint he rows and cols of the height map Z
hf_win_sz = round(0.5*spot_sz / mpp);   % half windows size
Zx = zeros(m, n);
Zy = zeros(m, n);

Z = remove_surface(X,Y,Z);

%% 1. Calculate the slope maps by sliding the window
for i = 1 : m
    for j = 1 : n
        rows = i - hf_win_sz : i + hf_win_sz;   % construct rows of the win
        cols = j - hf_win_sz : j + hf_win_sz;   % construct cols of the win
        
        rows = rows(rows>=1 & rows<=m); % ensure rows within limit
        cols = cols(cols>=1 & cols<=n); % ensure cols within limit
        
        Xwin = X(rows, cols);
        Ywin = Y(rows, cols);
        Zwin = Z(rows, cols);   % get the window's heights
        
        [~,~,f] = remove_surface(Xwin, Ywin, Zwin); % calculate the slopes
        Zx(i,j) = f(2); % obtain the slope x
        Zy(i,j) = f(3); % obtain the slope y
    end
end

end