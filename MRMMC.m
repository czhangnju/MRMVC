function [Graph] = MRMMC(X, param)
% X{i}: one view in d_i * n

alpha = param.alpha;
beta  = param.beta;

num_view = length(X);
[~, n] = size(X{1});

% Initialization
E = cell(1, num_view);
Z = E; D = E; XX = E; dist = E;
for iv = 1:num_view
    X{iv} = NormalizeFea(X{iv}, 0);
    E{iv} = zeros(size(X{iv}));
    Z{iv} = zeros(n, n);
    D{iv} = L2_distance_1(X{iv}, X{iv});
    XX{iv} = X{iv}'*X{iv};
    dist{iv} = L2_distance_1(X{iv}, X{iv});
end
Y1 = E;
Y2 = Z;
Y3 = Z;
Q = Z;
S = Z;


MAX_ITER = 30;
rho   = 1e-3;
theta = 1.5;
rho_max = 1e4;

for iter = 1:MAX_ITER
    iter;
    % E step
    for iv = 1:num_view
        E{iv} = prox_l21(X{iv}-X{iv}*Z{iv} + Y1{iv}/rho, 1/rho);
    end

    % Z step
    for iv = 1:num_view
        Z{iv} = (X{iv}'*X{iv}+2*eye(n))\(X{iv}'*(X{iv}-E{iv}+Y1{iv}/rho) + Q{iv} - Y2{iv}/rho + S{iv} - Y3{iv}/rho);
    end




     % Q step
    Z_tensor = cat(3, Z{:,:});
    Y2_tensor = cat(3, Y2{:,:});
    Zv = Z_tensor(:);
    Y2v = Y2_tensor(:);
   [Qv, ~] = wshrinkObj(Zv + 1/rho*Y2v, beta/rho, [n, n, num_view], 0, 3);
    Q_tensor = reshape(Qv, [n, n, num_view]);  
    for iv = 1:num_view
        Q{iv} = Q_tensor(:,:,iv);
    end

       % S step
    for iv = 1:num_view
        St     = Z{iv} + (Y3{iv} - alpha*dist{iv})/rho;
        St     = St - diag(diag(St));
        for ic = 1:n
            idx  = 1:n;
            idx(ic) = [];
            S{iv}(ic,idx) = EProjSimplex_new(St(ic,idx));          % 
        end
    end


    %
    for iv = 1:num_view
        Y1{iv} = Y1{iv} + rho*(X{iv} - X{iv}*Z{iv} -E{iv});
        Y2{iv} = Y2{iv} + rho*(Z{iv} - Q{iv});
        Y3{iv} = Y3{iv} + rho*(Z{iv} - S{iv});
    end
    rho = min(rho_max, theta*rho);
    
    epsilon = 1e-3;
    Err1 = []; Err2=[]; Err3=[];
    Loss1(iter) = 0; Loss2(iter) = 0; Loss3(iter) = 0;
    for iv = 1:num_view
        e1 = norm(X{iv} - X{iv}*Z{iv} -E{iv},'fro')^2;
        e2 = norm(Z{iv} - Q{iv}, 'fro')^2;
        e3 = norm(Z{iv} - S{iv}, 'fro')^2;
        Loss1(iter) = Loss1(iter) + e1;
        Loss2(iter) = Loss2(iter) + e2;
        Loss3(iter) = Loss3(iter) + e3;
    end
    if max(max(Loss1, Loss2), max(Loss3)) < epsilon
        break;
    end

end

Graph = 0;
for iv = 1:num_view
    Graph = Graph + (abs(Z{iv})+abs(Z{iv})')/2;
end
Graph = Graph/num_view;


end