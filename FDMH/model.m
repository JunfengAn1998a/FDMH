function [B_trn,alpha,P,R,Q] = model(param, data)

lambda = param.lambda;
beta = param.beta;
theta = param.theta;
eta = param.eta;
rho = param.rho;
rho0 = param.rho0;
miu = param.miu;
M = param.M;

indexTrain = data.indexTrain;
indexTest = data.indexTest;

gnd = data.gnd';
Y = gnd(:,indexTrain);

n = size(indexTrain,2);
r = param.bits;
p= size(data.X{1},1);
c = size(gnd,1);

X = zeros(p,n,M);
X_test = cell(1,M);
for i = 1:M
    X(:,:,i) = data.X{i}(:,indexTrain);
    X_test{i} = data.X{i}(:,indexTest);
end

%% Initialize
alpha = ones(length(X),1)/length(X);
Z = randn(r,p,M);
pi = randn(r,p,M);
P = randn(r,p,M);
V = randn(r, n);
B = randn(r,n);
B = B>0;B = B*2-1;
R = randn(c,r);
Q = randn(size(Y,1),r);
ub = ones(M,1);
lb = zeros(M,1);
Ialpha = ones(1,M);
chgl = 100;
%% -------------------Train-------------------------
while(chgl>=1e-7)
    %-------------alpha-STEP-----------------
    for t = 1:M
        Wour{:,:,t}=P(:,:,t)*X(:,:,t);
    end
    Hquad = zeros(M,M);
    for i = 1:M
        for j = 1:M
            Hquad(i,j) = trace(Wour{:,:,i}*Wour{:,:,j}');
        end
    end
    fquad = zeros(M,1);
    for i = 1:M
        fquad(i) = trace(Wour{:,:,i}*V');
    end
    options = optimoptions('quadprog','Display','off');
    alpha = quadprog(Hquad,-fquad,[],[],Ialpha,1,lb,ub,[],options);
    alpha = round(100*alpha) / 100;
    %-------------R-STEP-----------------
    [F1,~,F2] = svd(V*B','econ');
    R = F1*F2';
    %-------------P-STEP-----------------
    D = -V;
    for t = 1:M
        for j = 1:M
            if j == t
                continue;
            end
            D = D + alpha(j)*P(:,:,j)*X(:,:,j);
        end
        P(:,:,t) = (-2*alpha(t)*D*X(:,:,t)' + rho*Z(:,:,t) - pi(:,:,t)) / (2*(alpha(t)^2)*X(:,:,t)*X(:,:,t)'+rho*eye(p));
    end
    %-------------Z-STEP-----------------
    [Z,~,~] = prox_tnn(P+pi/rho,lambda/rho);
    %-------------B-STEP-----------------
    A = R*V;
    mu = median(A,2);
    A = bsxfun(@minus, A, mu);
    B = sign(A);
    %-------------Q-STEP-----------------
    Q = (theta*Y*V') / (eta*eye(r)+theta*V*(V'));
    %-------------V-STEP-----------------
    D = beta*R'*B + theta*Q'*Y;
    for t = 1:M
       D = D + alpha(t)*P(:,:,t)*X(:,:,t);
    end
    V = ((1+beta)*eye(r) + theta*(Q')*Q) \ D;
    %-------------pi,rho-STEP-----------------
    rho = min(rho*miu,rho0);
    pi = pi+rho*(P-Z);
    
    chgl = max(abs(Z(:)-P(:)));
end
B_trn = B>0;

