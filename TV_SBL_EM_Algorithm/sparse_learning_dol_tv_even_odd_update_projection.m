function [mu,dmu,k,gamma] = sparse_learning2_even_odd_update_projection(Phi,T,lambda,iters,flag1,flag2,flag3,beta_sbl,SNR,SigmaN)


% *** Control parameters ***
MIN_GAMMA       = -inf; %1e-12;  % 1e-16;
MIN_DMU         = 1/(10^(SNR/10))*1e-7;  %1e-12;
MAX_ITERS       = iters;
DISPLAY_FLAG    = flag3;     % Set to zero for no runtime screen printouts

% *** Initializations ***
[N M] = size(Phi);
[N L] = size(T);

if (~flag2)         gamma = ones(M,1);
else                gamma = flag2;  end;

keep_list = [1:M]';
m = length(keep_list);
mu = zeros(M,L);
dmu = 1;
k = 0;

loss_arr = zeros(iters,1);
tv_arr = zeros(iters,1);
L_arr = zeros(iters,1);

% Parameters
eps1 = 1e-11;
eps2 = 1e-17;

gamma_hat = zeros(M,1);
gamma_tilde = zeros(M,1);
gamma_bar = zeros(M,1);
gamma_max = zeros(M,1);
gamma_min = zeros(M,1);

S1 = 5*ones(M,1);
S2 = 5*ones(M,1);
S0 = 5*ones(M,1);
S_gamma = 5*ones(M,1);

F1 = zeros(M,1);
F2 = zeros(M,1);
F0 = zeros(M,1);
F00 = zeros(M,1);

beta_sbl_old = beta_sbl;


% *** Learning loop ***
while (1)
    
    % *** Prune things as hyperparameters go to zero ***
    if (min(gamma) < MIN_GAMMA )
        index = find(gamma > MIN_GAMMA);
        gamma = gamma(index);
        Phi = Phi(:,index);
        keep_list = keep_list(index);
        m = length(gamma);
        
        if (m == 0)   break;  end;
    end;
    
    
    % *** Compute new weights ***
    G = repmat(sqrt(gamma)',N,1);
    PhiG = Phi.*G;
    [U,S,V] = svd(PhiG,'econ');
    
    [d1,~] = size(S);
    if (d1 > 1)     diag_S = diag(S);
    else            diag_S = S(1);      end;
    
    U_scaled = U(:,1:min(N,m)).*repmat((diag_S./(diag_S.^2 + lambda + 1e-16))',N,1);
    Xi = G'.*(V*U_scaled');
    
    mu_old = mu;
    mu = Xi*T;
    
    % *** Update hyperparameters ***
    gamma_old = gamma;
    mu2_bar = sum(abs(mu).^2,2);
    
    if (flag1(1) == 0)
        % MacKay fixed-point SBL
        R_diag = real( (sum(Xi.'.*Phi)).' );
        gamma = mu2_bar./(L*R_diag);
        
    elseif (flag1(1) == 1)
        % Fast EM SBL
        R_diag = real( (sum(Xi.'.*Phi)).' );
        gamma = sqrt( gamma.*real(mu2_bar./(L*R_diag)) );
        
    elseif (flag1(1) == 2)
        % Traditional EM SBL
        PhiGsqr = PhiG.*G;
        Sigma_w_diag = real( gamma - ( sum(Xi.'.*PhiGsqr) ).' );
        gamma = mu2_bar/L + Sigma_w_diag;
         
    elseif (flag1(1) == 3)   % Markus
       

        
    elseif (flag1(1) == 4)        
        %% Sat 11.4. -- SBL-EM with Majorization 
        PhiGsqr = PhiG.*G;        
        Sigma_w_diag = real( gamma - ( sum(Xi.'.*PhiGsqr) ).' );
        eps0 = 1e-15;
        
        %figure(1); clf; hold on;
        
        if (beta_sbl(1)==0)
            gamma = mu2_bar/L + Sigma_w_diag;  % Traditional SBL-EM update 
            %gamma = max(0, sqrt( (mu2_bar/L + Sigma_w_diag)./( 1./(gamma + eps0) ) ) );
        else
            if(k<min(MAX_ITERS/2,200))
                beta_sbl = 1e-15*ones(M,1);
            else
                beta_sbl = beta_sbl_old;
            end
            
            % Universal parameters
            E_i = real(mu2_bar/L + Sigma_w_diag);

            
            %%%% Parallel update step for log %%%%
            
%             % Unconstrained optimization update: With Majorization
%             q_i = 1./(real(gamma) + eps0);
%             t_i = beta_sbl./(real(gamma) + eps1);
%             r_i = beta_sbl./q_i;
% %             t_i(1) = t_i(1)/2;
% %             t_i(M) = t_i(M)/2;
% %             r_i(1) = r_i(1)/2;
% %             r_i(M) = r_i(M)/2;
%             
%             alpha_hat = real( max(0,sqrt(E_i./(q_i + 2*t_i))) );
%             %gamma_hat = real( max(0,sqrt(E_i./q_i + r_i.^2) - r_i) );
%             alpha_tilde = real( max(0,sqrt(E_i./(q_i))) );
%             alpha_bar = real( max(0,sqrt(E_i./q_i + r_i.^2) + r_i) );


            % Unconstrained optimization without majorization
            beta_sbl(1) = beta_sbl(1)/2;
            beta_sbl(M) = beta_sbl(M)/2;
            a_i = sqrt(1 + ( 4*(1+2*beta_sbl).*E_i*eps1./(E_i - eps1).^2 ) );
            b_i = sqrt(1 + ( 4*(1-2*beta_sbl).*E_i*eps1./(E_i - eps1).^2 ) );
            
            alpha_hat = real(max(0,E_i./(1 + 2*beta_sbl )));
            %alpha_hat = real(max(0,(E_i - eps1).*(1 + a_i)./(1 + 2*beta_sbl )/2 ) );
            alpha_tilde = real(max(0,E_i ) );
            %alpha_bar = real(max(0,E_i./(1 - 2*beta_sbl )));
            alpha_bar = real(max(0,E_i./(max(eps2, 1 - 2*beta_sbl ) ) ) );
            %alpha_bar = real(max(0,(E_i - eps1).*(1 + b_i)./(1 - 2*beta_sbl )/2 ) );
            
%             if(1-2*beta_sbl(3)<0)
%                gamma_bar = real(max(0,E_i./eps2 ) ); 
%             end
            
            
            for even_odd_index = 1:2
                
                gamma_hat_old = gamma_hat;
                gamma_tilde_old = gamma_tilde;
                gamma_bar_old = gamma_bar;
                
                flag_hat = zeros(M,1);
                flag_tilde = zeros(M,1);
                flag_bar = zeros(M,1);
                
                %%%% Max and min of neighbours %%%%
                gamma_max(2:end-1) = max([gamma(1:end-2),gamma(3:end)],[],2);
                gamma_max(1) = gamma(2);
                gamma_max(M) = gamma(M-1);
                gamma_min(2:end-1) = min([gamma(1:end-2),gamma(3:end)],[],2);
                gamma_min(1) = gamma(2);
                gamma_min(M) = gamma(M-1);
                
                %%%% Even odd update assignment %%%%
                flag_even_odd = zeros(M,1);
                if(even_odd_index==1)
                    flag_even_odd(2:2:end)=1;
                else
                    flag_even_odd(1:2:end)=1;
                end
            
                %%%% Sign gain function %%%%
                %if(even_odd_index==1)
                    % S1
                    S1(2:M-1) = sign(gamma_hat_old(2:M-1) - gamma(1:M-2)) + sign(gamma_hat_old(2:M-1) - gamma(3:M));
                    S1(1) = sign(gamma_hat_old(1) - gamma(2));
                    S1(M) = sign(gamma_hat_old(M) - gamma(M-1));

                    % S0
                    S0(2:M-1) = sign(gamma_tilde_old(2:M-1) - gamma(1:M-2)) + sign(gamma_tilde_old(2:M-1) - gamma(3:M));
                    S0(1) = sign(gamma_tilde_old(1) - gamma(2));
                    S0(M) = sign(gamma_tilde_old(M) - gamma(M-1));

                    % S2
                    S2(2:M-1) = sign(gamma_bar_old(2:M-1) - gamma(1:M-2)) + sign(gamma_bar_old(2:M-1) - gamma(3:M));
                    S2(1) = sign(gamma_bar_old(1) - gamma(2));
                    S2(M) = sign(gamma_bar_old(M) - gamma(M-1));
                    
                    % S_gamma
                    S_gamma(2:M-1) = sign(gamma(2:M-1) - gamma(1:M-2)) + sign(gamma(2:M-1) - gamma(3:M));
                    S_gamma(1) = sign(gamma(1) - gamma(2));
                    S_gamma(M) = sign(gamma(M) - gamma(M-1));

                    S1 = real(S1);
                    S2 = real(S2);
                    S0 = real(S0);
                %end


                %%%% Coupling block %%%%
                % First set key values
                key_ind = (S1==2 & flag_even_odd==1);
                gamma(key_ind) = gamma_hat_old(key_ind);
                flag_hat(key_ind) = 1;
                if(S1(1)==1 && flag_even_odd(1)==1)
                    gamma(1) = gamma_hat_old(1);
                    flag_hat(1) = 1;
                end
                if(S1(M)==1 && flag_even_odd(M)==1)
                    gamma(M) = gamma_hat_old(M);
                    flag_hat(M) = 1;
                end

                if(key_ind(1)==0)
                    gamma(1) = gamma_tilde_old(1);
                    flag_tilde(1) = 1;
                    if(flag_hat(1)==1)
                        flag_hat(1) = 0;
                    end
                end
                if(key_ind(M)==0)
                    gamma(M) = gamma_tilde_old(M);
                    flag_tilde(M) = 1;
                    if(flag_hat(M)==1)
                        flag_hat(m) = 0;
                    end
                end
                key_ind = (S0==0 & flag_even_odd==1);
                gamma(key_ind) = gamma_tilde_old(key_ind);
                flag_tilde(key_ind) = 1;

                key_ind = (S2==-2 & flag_even_odd==1);
                gamma(key_ind) = gamma_bar_old(key_ind);
                flag_bar(key_ind) = 1;
                if(S2(1)==-1 && flag_even_odd(1)==1)
                    gamma(1) = gamma_bar_old(1);
                    flag_bar(1) = 1;
                    if(flag_tilde(1)==1)
                        flag_tilde(1) = 0;
                    end
                end
                if(S2(M)==-1 && flag_even_odd(M)==1)
                    gamma(M) = gamma_bar_old(M);
                    flag_bar(M) = 1;
                    if(flag_tilde(M)==1)
                        flag_tilde(M) = 0;
                    end
                end
                
                % Inconsistencies
                index_doubt = flag_even_odd==1;
                %index_doubt = (flag_hat+flag_tilde+flag_bar)==0 & flag_even_odd==1;
                

                %%%%%%%%% ORIGINAL METHOD OF LIKELIHOODS %%%%%%%%%
                % Projection
                %if(even_odd_index == 26)
                    gamma_hat(2:end-1) = max([alpha_hat(2:end-1),gamma_max(2:end-1)],[],2);
                    gamma_hat(1) = max([alpha_hat(1),gamma_max(1)],[],2);
                    gamma_hat(end) = max([alpha_hat(end),gamma_max(end)],[],2);

                    gamma_tilde(2:end-1) = min([gamma_max(2:end-1),max([alpha_tilde(2:end-1),gamma_min(2:end-1)],[],2)],[],2);
                    gamma_tilde(1) = max([alpha_tilde(1),gamma(2)],[],2);
                    gamma_tilde(end) = min([alpha_tilde(end),gamma(end-1)],[],2);

                    gamma_bar(2:end-1) = min([alpha_bar(2:end-1),gamma_min(2:end-1)],[],2);
                    gamma_bar(1) = min([alpha_bar(1),gamma_min(1)],[],2);
                    gamma_bar(end) = min([alpha_bar(end),gamma_min(end)],[],2);
                %end
                
                
                % Likelihood
                %f_i = E_i./gamma_hat + q_i.*gamma_hat;
                f_i = E_i./gamma_hat + log(gamma_hat + eps0);
                %F1(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(2*log(gamma_hat(2:end-1) + eps1) - log(gamma(1:end-2) + eps1) - log(gamma(3:end) + eps1));
                F1(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(log(gamma_hat(2:end-1) + eps1) - log(gamma(3:end) + eps1)) + beta_sbl(2:end-1).*abs(log(gamma_hat(2:end-1) + eps1) - log(gamma(1:end-2) + eps1));
                F1(1) = f_i(1) + beta_sbl(1).*abs(log(gamma_hat(1) + eps1) - log(gamma(2) + eps1) );
                F1(M) = f_i(M) + beta_sbl(M).*abs(log(gamma_hat(M) + eps1) - log(gamma(M-1) + eps1) );

                %f_i = E_i./gamma_bar + q_i.*gamma_bar;
                f_i = E_i./gamma_bar + log(gamma_bar + eps0);
                %F2(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(2*log(gamma_bar(2:end-1) + eps1) - log(gamma(1:end-2) + eps1) - log(gamma(3:end) + eps1));
                F2(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(log(gamma_bar(2:end-1) + eps1) - log(gamma(3:end) + eps1)) + beta_sbl(2:end-1).*abs(log(gamma_bar(2:end-1) + eps1) - log(gamma(1:end-2) + eps1));
                F2(1) = f_i(1) + beta_sbl(1).*abs(log(gamma_bar(1) + eps1) - log(gamma(2) + eps1) );
                F2(M) = f_i(M) + beta_sbl(M).*abs(log(gamma_bar(M) + eps1) - log(gamma(M-1) + eps1) );

                %f_i = E_i./gamma_tilde + q_i.*gamma_tilde;
                f_i = E_i./gamma_tilde + log(gamma_tilde + eps0);
                %F0(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(log(gamma(1:end-2) + eps1) - log(gamma(3:end) + eps1));
                F0(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(log(gamma_tilde(2:end-1) + eps1) - log(gamma(3:end) + eps1)) + beta_sbl(2:end-1).*abs(log(gamma_tilde(2:end-1) + eps1) - log(gamma(1:end-2) + eps1));
                F0(1) = f_i(1) + beta_sbl(1).*abs(log(gamma_tilde(1) + eps1) - log(gamma(2) + eps1) );
                F0(M) = f_i(M) + beta_sbl(M).*abs(log(gamma_tilde(M) + eps1) - log(gamma(M-1) + eps1) );
                
                %f_i = E_i./gamma + q_i.*gamma;
                f_i = E_i./gamma + log(gamma + eps0);
                %F0(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(log(gamma(1:end-2) + eps1) - log(gamma(3:end) + eps1));
                F00(2:end-1) = f_i(2:end-1) + beta_sbl(2:end-1).*abs(log(gamma(2:end-1) + eps1) - log(gamma(3:end) + eps1)) + beta_sbl(2:end-1).*abs(log(gamma(2:end-1) + eps1) - log(gamma(1:end-2) + eps1));
                F00(1) = f_i(1) + beta_sbl(1).*abs(log(gamma(1) + eps1) - log(gamma(2) + eps1) );
                F00(M) = f_i(M) + beta_sbl(M).*abs(log(gamma(M) + eps1) - log(gamma(M-1) + eps1) );
                
                % Doubtful values
                gamma_1 = gamma_hat(index_doubt);
                gamma_0 = gamma_tilde(index_doubt);
                gamma_2 = gamma_bar(index_doubt);
                gamma_temp = gamma(index_doubt);

                % Likelihood doubt
                F1_doubt = F1(index_doubt);
                F2_doubt = F2(index_doubt);
                F0_doubt = F0(index_doubt);
                %F00_doubt = F00(index_doubt);

                % Minimizer
                [~,ind] = min([F1_doubt,F0_doubt,F2_doubt],[],2);

                % Allocation
                gamma_temp(ind==1) = gamma_1(ind==1);
                gamma_temp(ind==2) = gamma_0(ind==2);
                gamma_temp(ind==3) = gamma_2(ind==3);

                gamma(index_doubt) = gamma_temp;
            end
        end
        
    else
        % FOCUSS
        p = flag1(2);
        gamma = (mu2_bar/L).^(1-p/2);
    end;
    
    % *** Check stopping conditions, etc. ***
    k = k+1;
    
    % Likelihood evalutation
%     sig_y = lambda*eye(N) + Phi*diag(real(gamma))*(Phi');
%     L_arr(k) = L*log(det(sig_y));
%     for meas_num = 1:L
%         T_meas = T(:,meas_num);
%         L_arr(k) = L_arr(k) + ((T_meas')/sig_y)*T_meas;
%     end
%     L_arr(k) = L_arr(k) + sum(beta_sbl(2:end).*abs((sqrt(2*real(gamma(2:end))+ eps5)) - (sqrt(2*real(gamma(1:end-1))+ eps5))));
%     L_arr(k) = real(L_arr(k)); 
    
    
    
    if (DISPLAY_FLAG) disp(['iters: ',num2str(k),'   num coeffs: ',num2str(m), ...
            '   gamma change: ',num2str(max(abs(gamma - gamma_old)))]); end;
    if (k >= MAX_ITERS) break;  end;
    
    if (size(mu) == size(mu_old))
        dmu = max(max(abs(mu_old - mu)));
        %if (dmu < MIN_DMU)  break;  end;
    end;
    
     

            
end;

% *** Expand weights, hyperparameters ***
temp = zeros(M,1);
if (m > 0) temp(keep_list,1) = gamma;  end;
gamma = temp;

temp = zeros(M,L);
if (m > 0) temp(keep_list,:) = mu;  end;
mu = temp;

%fprintf('\n   EM ites: %d/%d\n', k , iters)

return;
