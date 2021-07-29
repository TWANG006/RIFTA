function F = zero_pad(F0,mm,nn)
%expand f0 to [m n]
%this function can be realized by padarray, but is slower

%size f0
[m, n]=size(F0); 

%generate a larger matrix with size [mm nn]
F=zeros(mm,nn);

%copy original data
F(1:m,1:n)=F0;

end
