function [F_AR] = feature_Ac(a,b,comp_sel)

% Matrix index
    
Test_R = load('tmp_/R_txt.txt');
Test_R = reshape(Test_R, [b,a])';

F_AR = zeros(size(Test_R,1),size(Test_R,2));
Test_W2 = zeros(size(Test_R,1),size(Test_R,2));
Test_C2 = zeros(size(Test_R,1),size(Test_R,2));

A_R_P = zeros(size(Test_R,1),1);  
A_R_S = zeros(size(Test_R,1),1); 


filename = ['base/Net_def.txt'];
filename1 =['base/R_list.txt'];
filename2 =['base/A_norm.txt'];
filename3 =['base/W_norm.txt'];
filename4 =['base/C_norm.txt'];


delimiterIn = ' ';
Net = importdata(filename,delimiterIn);

num_fc = Net(:,7)-Net(:,8);
num_fc = sum(num_fc == 0);
if comp_sel == 2
    L = size(Net,1)
else
    L = size(Net,1)-num_fc
end

if comp_sel == 0
    Net = Net(1:L,:);
end

W_orig = sum(Net(:,7)); 
C_orig = sum(Net(:,8)); 
FC_cost = sum(Net(end-num_fc+1:end,7));

a = Net(:,8)./Net(:,2);
w = Net(:,7)./Net(:,2);


R1 = importdata(filename1,delimiterIn);
R1 = R1(:,1:L);
R = [zeros(1,L) ; R1 ];

A1 = importdata(filename2,delimiterIn);
A1 = A1(:,1:L);
A = [zeros(1,L) ; A1 ];

W1 = importdata(filename3,delimiterIn);
W1 = W1(:,1:L);
W = [zeros(1,L) ; W1 ];

C1 = importdata(filename4,delimiterIn);
C1 = C1(:,1:L);
C = [zeros(1,L) ; C1 ];


R_max = R(end,:);
R_norm = R(:,:)./R_max;
 
% Accuracy

A_norm = A(:,:)./A(end,:);
a_max = A(end,1);

% Weight

Wmax = W(end,:);
W_norm = W(:,:)./Wmax;

% FLOPs

Cmax = C(end,:)
C_norm = C(:,:)./Cmax;


L = size(A_norm,2);
for i=1:L

    [~, ind] = unique(A_norm(:,i));
    A_norm_tmp{i} = A_norm(ind,i);
    R_norm_tmp{i} = R_norm(ind,i);

    [~, ind] = unique(R_norm_tmp{i});
    A_norm_tmp{i} = A_norm_tmp{i}(ind);
    R_norm_tmp{i} = R_norm_tmp{i}(ind);

	f_AR{i} = interp1(R_norm_tmp{i}, A_norm_tmp{i},'pchip','pp');

end

Test_C = Test_R.*a';
Test_W = Test_R.*w';

W_acc = sum(Test_W,2)./(C_orig);
C_acc = sum(Test_C,2)./(W_orig);

Test_R2 = (Test_R)./R_max(1:L); 
for i=1:L
	F_AR(:,i) = ppval(f_AR{i},Test_R2(:,i));
end
A_R_P = prod(F_AR,2);
A_R_S = sum(F_AR,2);

A_R_S_Cp = prod(F_AR,2).*C_acc;
A_R_S_Cd = prod(F_AR,2)./C_acc;
out = [A_R_P, A_R_S_Cp, A_R_S_Cd];

save(['tmp_/MATLAB_feature.txt'], 'out', '-ascii');

end
