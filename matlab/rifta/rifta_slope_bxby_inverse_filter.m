function Tdg = rifta_slope_bxby_inverse_filter(...
    gamma,...
    FZxy,...
    FBxy,...
    Coeffs...
    )

[Ny, Nx] = size(FZxy);

alpha = Coeffs ./ gamma;
beta = gamma ./ Coeffs;
beta(1, 1) = 0;
sFBxy = FBxy .* (abs(FBxy .* beta) > 1) + alpha .* (abs(FBxy .* beta) <= 1);  % remove 0's
iFBxy = 1./ sFBxy;
iFBxy(1, 1) = 0;

Tdg = real(ifft2(iFBxy .* FZxy));
Tdg = Tdg(1 : Ny/2, 1 : Nx/2);

end