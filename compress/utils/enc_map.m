
function [R_Amax] = enc_map(Ctarget, space, comp_sel, C_flag, conv1_half)

filename = ['base/Net_def.txt'];
filename1 =['base/R_list.txt'];
filename2 =['base/A_norm.txt'];
filename3 =['base/W_norm.txt'];
filename4 =['base/C_norm.txt'];


delimiterIn = ' ';
Net = importdata(filename,delimiterIn);

num_fc = Net(:,7)-Net(:,8);
num_fc = sum(num_fc == 0)
if comp_sel == 2
    L = size(Net,1)
else
    L = size(Net,1)-num_fc
end

if comp_sel == 0
    Net = Net(1:L,:)
end

a = Net(:,8)./Net(:,2);
w = Net(:,7)./Net(:,2);

W_orig = sum(Net(:,7)); 
C_orig = sum(Net(:,8)); 
FC_cost = sum(Net(end-num_fc+1:end,7));


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
 

R_max = R(end,:);
R_norm = R(:,:)./R_max;
 
% Accuracy
A_norm = A(:,:)./A(end,:)
a_max = A(end,1);

% Weight
Wmax = W(end,:)
W_norm = W(:,:)./Wmax;

% FLOPs
Cmax = C(end,:)
C_norm = C(:,:)./Cmax;

n = size(R_norm,1);
m = size(R_norm,2);


%% Generation Cost Function - from A

a_norm = [0.1:0.01:1];
max_rank = [];
for i=1:size(A_norm,2)

    A_norm_tmp{i} = A_norm(:,i);
    R_norm_tmp{i} = R_norm(:,i);

    A_mask = A_norm_tmp{i} < 1;
    A_idx(i) = sum(A_mask)+1;

    A_norm_tmp{i} = A_norm_tmp{i}(1:A_idx(i));
    R_norm_tmp{i} = R_norm_tmp{i}(1:A_idx(i));

    [~, ind] = unique(A_norm_tmp{i});
    A_norm_tmp{i} = A_norm_tmp{i}(ind);
    R_norm_tmp{i} = R_norm_tmp{i}(ind);

    [~, ind] = unique(R_norm_tmp{i});
    A_norm_tmp{i} = A_norm_tmp{i}(ind);
    R_norm_tmp{i} = R_norm_tmp{i}(ind);


    diff_a = [];
    A_idx(i) = length(A_norm_tmp{i});
    for k=1:A_idx(i)-1
        diff_a(k) = A_norm_tmp{i}(k+1) - A_norm_tmp{i}(k);
    end
    diff_mask = diff_a<=0;
    diff_num = sum(diff_mask);
    tmp_a = A_norm_tmp{i}(2:end);
    tmp_a = tmp_a(~diff_mask); % +1 remove
    tmp_r = R_norm_tmp{i}(2:end);
    tmp_r = tmp_r(~diff_mask); % +1 remove
    A_norm_tmp{i} = [A_norm_tmp{i}(1);tmp_a];
    R_norm_tmp{i} = [R_norm_tmp{i}(1);tmp_r];

    A_idx(i) = A_idx(i) - diff_num;
    max_rank(i) = R_norm_tmp{i}(end);
end


for n=1:size(a_norm,2)
    for i=1:size(A_norm,2)
       
		x = A_norm_tmp{i};
		y = R_norm_tmp{i};

        pr_int(i) = round(interp1(x, y, a_norm(n),'pchip')*R(end,i));
    end
    r_min_vbmf(n,:) = pr_int;
end



%% Cost Calculation - from A
Test_R = r_min_vbmf;
Test_C = Test_R.*a';
Test_W = Test_R.*w';

if conv1_half == 1
    Test_W(:,1) = 0;
    Test_C(:,1) = 0;
end

W_acc = sum(Test_W,2)./W_orig;
C_acc = sum(Test_C,2)./C_orig;


%% Refinement of complexity 
if Ctarget > max(C_acc)
    for n=1:size(a_norm,2)
        for i=1:size(A_norm,2)
            R_norm_tmp{i}(end)=1;
            x = A_norm_tmp{i};
            y = R_norm_tmp{i};
            pr_int(i) = round(interp1(x, y, a_norm(n),'pchip')*R(end,i));
        end
        r_min_vbmf(n,:) = pr_int;
    end

    Test_R = r_min_vbmf;
    Test_C = Test_R.*a';
    Test_W = Test_R.*w';

    if conv1_half == 1
        Test_W(:,1) = 0;
        Test_C(:,1) = 0;
    end

    W_acc = sum(Test_W,2)./W_orig;
    C_acc = sum(Test_C,2)./C_orig;
end

% C_flag - 0 : Weight / 1 : FLOPs
if C_flag == 0
    C_acc = W_acc;
end

if conv1_half == 1
    if C_flag == 1
        Ctarget = Ctarget - Net(1,8)/Net(1,2)*round(Net(1,2)/2)/C_orig
    elseif C_flag == 0
        Ctarget = Ctarget - Net(1,7)/Net(1,2)*round(Net(1,2)/2)/W_orig
    end
end


%% Interpolate function
[~, ind] = unique(a_norm);
a_norm = a_norm(ind);
C_acc = C_acc(ind);

[~, ind] = unique(C_acc);
a_norm = a_norm(ind);
C_acc = C_acc(ind);

a_norm = [0 a_norm];
C_acc = [0; C_acc];

[C_acc_, W_acc_, Rmax] = Ro_calculation(Ctarget, space, C_acc, a_norm, A_norm, A_norm_tmp, R_norm_tmp, A_idx, R, a, w, conv1_half, C_orig, W_orig, C_flag)

C_acc_out(1) = C_acc_
W_acc_out(1) = W_acc_

[C_acc_, W_acc_, Rmin] = Ro_calculation(Ctarget, -space, C_acc, a_norm, A_norm, A_norm_tmp, R_norm_tmp, A_idx, R, a, w, conv1_half, C_orig, W_orig, C_flag)

C_acc_out(2) = C_acc_
W_acc_out(2) = W_acc_

if conv1_half == 1
    Rmax(1) = round(Net(1,2)/2);
    Rmin(1) = round(Net(1,2)/2);
    C_acc_out = C_acc_out + Net(1,8)/Net(1,2)*Rmax(1)/C_orig;
    W_acc_out = W_acc_out + Net(1,7)/Net(1,2)*Rmax(1)/W_orig;
end

Rmax
Rmin
C_acc_out
W_acc_out

% file write
Set = [Rmax, Rmin, C_acc_out, W_acc_out];
save(['tmp_/MATLAB_result.txt'], 'Set', '-ascii');

disp('~~~~~~~~~~~~~~~~~~~~~~')    
disp('~~ Boundary Gen End ~~ ');
disp('~~~~~~~~~~~~~~~~~~~~~~')    



end



function [C_acc_o, W_acc_o, Rmax] = Ro_calculation(Ctarget, space, C_acc, a_norm, A_norm, A_norm_tmp, R_norm_tmp, A_idx, R, a, w, conv1_half, C_orig, W_orig, C_flag)

    C_acc_ = C_acc;
    end_flag = 1
    delta = 0;
    k = 0;
    while(end_flag>0)
        Amax = interp1(C_acc_,a_norm,Ctarget+space+delta,'pchip'); 
        if Ctarget+space+delta > 1
            Amax = 1.0;
        end
        acc_min_idx = sum(a_norm<0.9);

        L = size(A_norm,2);
        for i=1:L    
            x = A_norm_tmp{i}(1:A_idx(i));
            y = R_norm_tmp{i}(1:A_idx(i));
            Rmax(i) = max(interp1(x, y, Amax,'pchip'),0); 
        end
        Rmax = round(Rmax.*R(end,1:L));
        Test_R = Rmax;
        W_acc = [];
        C_acc = [];

        Test_C = Test_R.*a';
        Test_W = Test_R.*w';


        if conv1_half == 1
            Test_W(1) = 0;
            Test_C(1) = 0;
        end


        W_acc = sum(Test_W)./W_orig;
        C_acc = sum(Test_C)./C_orig;

        if C_flag == 0
            C_acc = W_acc;
        end

        if C_acc <= Ctarget+space
            delta = delta+0.001;
        else 
            delta = delta-0.001;
        end
        k = k+ 1;

        if (C_acc < (Ctarget +space + 0.001)) && ((Ctarget +space - 0.001) < C_acc)
            flag_end = 0;
        end
        if k > 500
            flag_end = 0;
            break;
        end
    end
    C_acc_o = C_acc;
    W_acc_o = W_acc;
end

