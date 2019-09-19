% Create a 6x8x10 random tensor
IM = randn(6,8,10);
% patch size
ps = [2,3,4];

% Learn dictionaries for modes 1, 2 and 3 with sizes (6x2), (8,3) and (10,4), respectively:
disp('Learning dictionaries')
D0 = DictLearn3D(IM,ps,0.1,2.0);
disp('Dictionaries leaned from data OK')

% Generate a mask for 10% data missing
Mask  = ones(size(IM));
indices = randperm(prod(size(IM)));
Mask(indices(1:round(prod(size(IM))*0.1))) = 0;



% sparsity
sparsity = 0.1;

% Call CesarTC completion algorithm
disp('Completing tensor')
[IMrec] = CesarTC(IM,Mask,ps,D0,sparsity);





