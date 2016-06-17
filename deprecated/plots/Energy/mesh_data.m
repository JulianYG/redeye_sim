cd ../Raw
mesh1 = csvread('goog_1P_mesh.csv',1,1);
mesh2 = csvread('goog_2P_mesh.csv',1,1);
mesh3 = csvread('goog_neg3P_mesh.csv',1,1);
mesh4 = csvread('goog_3P_mesh.csv',1,1);
mesh5 = csvread('goog_4P_mesh.csv',1,1);
cd ../Energy

q_SNR_range = [20, 26, 32, 38, 44, 50, 56];
q_SNR = [q_SNR_range,q_SNR_range,q_SNR_range,q_SNR_range,q_SNR_range,...
    q_SNR_range,q_SNR_range];

q = mesh1(:,8);
q_unscaled = q./(4*35.0324);

mesh1(:,10) = q_SNR;
mesh2(:,13) = q_SNR;
mesh3(:,19) = q_SNR;
mesh4(:,25) = q_SNR;
mesh5(:,31) = q_SNR;
mesh1(:,2) = q_unscaled;
mesh2(:,2) = q_unscaled;
mesh3(:,2) = q_unscaled;
mesh4(:,2) = q_unscaled;
mesh5(:,2) = q_unscaled;

%1st column is top 5 accuracy 
%4th column is unscaled G signal amplitude
%2nd column is unscaled Q signal amplitude
%second to last column is G_SNR
%last column is Q_SNR
