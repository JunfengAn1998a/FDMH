function [B] = hashFunction(X,alpha,P,R)
V = zeros(size(P,1),size(X{1},2));
for i = 1:size(X,2)
    V = V + alpha(i) * P(:,:,i) * X{i};
end
B = R*V;
B = B > 0;
