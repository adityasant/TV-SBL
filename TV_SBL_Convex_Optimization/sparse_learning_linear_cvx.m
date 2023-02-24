function [x_sbl,gamma] = sparse_learning_linear_cvx(A,Y,gamma,ite_sbl_max,beta_sbl,SigmaN,Diff_tol_sbl)
    [M N] = size(A);
    [M L] = size(Y);
    
    R = zeros(M,M,L);
    A_mm = zeros(M,N,L);
    %SigmaY_inv = zeros(M,M,L);
    for m=1:L
        R(:,:,m) = sqrtm(Y(:,m)*Y(:,m)');
        R(:,:,m) = 0.5 * ( R(:,:,m) + R(:,:,m)' ); % Ensure symmetry
        A_mm(:,:,m) = A;
    end
 
    %% SBL-MM | convex optimization
    for k=1:ite_sbl_max
        gamma_prev = gamma;
        SigmaY_inv = inv( SigmaN+A*diag(gamma_prev)*A' );
 
        cvx_begin sdp
        cvx_solver sedumi
        cvx_precision default
        cvx_quiet true
        variable gamma_cvx(N,1)
        variable Z_cvx(M,M,L) symmetric
        z_tv_cvx = 0;
        Z_sum = 0;
        LogDet_sum = L * trace( real( SigmaY_inv * ( A_mm(:,:,1)*diag(gamma_cvx)*A_mm(:,:,1)' ) ) );
        for m=1:L
            %m0 = ceil(m/MeasBlock);
            Z_sum = Z_sum + trace( Z_cvx(:,:,m) );
            %LogDet_sum = LogDet_sum + ...
            %    L * trace( real( SigmaY_inv(:,:,m) * ( SigmaN+A(:,:,m)*diag(gamma_cvx)*A(:,:,m)' ) ) );
        end
        for n = 2:N
            z_tv_cvx = z_tv_cvx + beta_sbl * abs( gamma_cvx(n) - gamma_cvx(n-1) );
        end
        minimize( LogDet_sum + Z_sum + z_tv_cvx )
        subject to
            gamma_cvx >= 0;
            for m=1:L
                %m0 = ceil(m/MeasBlock);
                [Z_cvx(:,:,m), R(:,:,m)';
                    R(:,:,m), SigmaN+A_mm(:,:,1)*diag(gamma_cvx)*A_mm(:,:,1)'] >= 0;
            end
        cvx_end
        gamma = gamma_cvx;
 
        x_sbl = zeros(N,L);
        for m=1:L
            SigmaY_inv = inv( SigmaN+A_mm(:,:,m)*diag(gamma)*A_mm(:,:,m)' );
            %SigmaY_inv = 0.5 * ( SigmaY_inv + SigmaY_inv ); % Ensure symmetry
            x_sbl(:,m) = diag(gamma)*A_mm(:,:,m)'*SigmaY_inv*Y(:,m);
        end
 
 
        if ( norm(gamma - gamma_prev, 'fro') / norm(gamma, 'fro') <= Diff_tol_sbl  )
            break;
        end
    end % <-- k
    
end
