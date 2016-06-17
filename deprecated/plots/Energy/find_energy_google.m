gvary=1;

% gvary=11; %one for gvary, 0 from qvary
file_name = 'googlenet_data_stats.csv';
kernel_name = 'googlenet_kernel_stats.csv';

SNR_qu_phase =[35.0324,35.6255,37.1505,25.8848,25.2142,43.4403]; %quantization scale...
...for each P (1P, 2P, 3P, 4P, 5P)
SNR_ga_phase = 79.99;
    
data = csvread(file_name,0,5);
output_d=data(:,1);
output_h=data(:,2);
output_w=data(:,3);

kernel = csvread(kernel_name,0,5);
kernel_h=kernel(:,1);
kernel_w=kernel(:,2);
kernel_d=kernel(:,3);

%1 PHASE

cd ../Raw
data_ga = csvread('hypercube_1P_goog_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_1P_goog_qvary_SNR.csv',1,1);
cd ../Energy

if gvary==1
    acc5_1= data_ga(:,1);
    g = data_ga(:,4);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    q = data_ga(:,8);
else
    acc5_1= data_qu(:,1);
    g = data_qu(:,4);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    q = data_qu(:,8);
end 

output_l = [2,4,4];
kernel_l = [1,0,0];

energy_1 = zeros(length(g1),1);
energy = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy = [unit_energy_g1, unit_energy_g2];

for i = 1:length(output_l)-1
    if kernel_l(i) == 0 
        energy_1 = energy_1 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
            .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
            + ones(length(g1),1) .* 9.096e-12;

    else
        energy_1 = energy_1 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
            kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
            * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
            * output_w(output_l(i),1) * output_d(output_l(i),1);
    end
end

%   bits_qu = log(128./q)/log(2) %how julian had it set up
q_unscaled = q./(4*SNR_qu_phase(1));
SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
bits_qu = (SNR_qu - 1.76)./6.02;

energy_q=0;
energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
num_entries = output_h(output_l(length(output_l)),1)*...
     output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
energy_q_1 = energy_q*num_entries;
energy_tot_1 = energy_1 + energy_q_1;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;
    
%2 PHASE

cd ../Raw
data_ga = csvread('hypercube_2P_goog_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_2P_goog_qvary_SNR.csv',1,1);
cd ../Energy

data_qu = data_qu(1:50,:);
data_ga = data_ga(1:50,:);

if gvary==1
    acc5_2= data_ga(:,1);
    g = data_ga(:,4);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7)
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    g5 = data_ga(:,10);
    q = data_ga(:,13);
else
    acc5_2= data_ga(:,1);
    g = data_qu(:,4);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    g5 = data_qu(:,10);
    q = data_qu(:,11);
end 

output_l = [2,4,5,6,7,8]; %last number is for q
kernel_l = [1,0,2,3,0,0];

energy = zeros(length(g1),1);
energy_2 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4, unit_energy_g5];

for i = 1:length(output_l)-1
    if kernel_l(i) == 0 
        energy_2 = energy_2 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
            .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
            + ones(length(g1),1) .* 9.096e-12;

    else
        energy_2 = energy_2 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
            kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
            * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
            * output_w(output_l(i),1) * output_d(output_l(i),1);
    end
end

q_unscaled = q./(4*SNR_qu_phase(2));
SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
bits_qu = (SNR_qu - 1.76)./6.02;

energy_q=0;
energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
num_entries = output_h(output_l(length(output_l)),1)*...
     output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
energy_q_2 = energy_q*num_entries;
energy_tot_2 = energy_2 + energy_q_2;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;

%neg3 PHASE 

cd ../Raw
data_ga = csvread('hypercube_neg3P_goog_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_neg3P_goog_qvary_SNR.csv',1,1);
cd ../Energy

data_qu = data_qu(1:50,:);
data_ga = data_ga(1:50,:);

if gvary==1
    acc5_neg3= data_ga(:,1);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    g5 = data_ga(:,10);
    g6 = data_ga(:,11);
    g7 = data_ga(:,12);
    g8 = data_ga(:,13);
    g9 = data_ga(:,14);
    g10 = data_ga(:,15);
    g11 = data_ga(:,16);
    q = data_ga(:,17);
    
else
    acc5_neg3= data_qu(:,1);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    g5 = data_qu(:,10);
    g6 = data_qu(:,11);
    g7 = data_qu(:,12);
    g8 = data_qu(:,13);
    g9 = data_qu(:,14);
    g10 = data_qu(:,15);
    g11 = data_qu(:,16);
    q = data_qu(:,17);
end 

output_l = [2,4,5,6,7,13,14,15,16,17,19,20]; %last number is for q
kernel_l = [1,0,2,3,0,4,5,6,7,8,9,0];

energy = zeros(length(g1),1);
energy_neg3 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
unit_energy_g6 = 3.424e-15*ones(length(g1),1) ./ (g6 .* g6);
unit_energy_g7 = 3.424e-15*ones(length(g1),1) ./ (g7 .* g7);
unit_energy_g8 = 3.424e-15*ones(length(g1),1) ./ (g8 .* g8);
unit_energy_g9 = 3.424e-15*ones(length(g1),1) ./ (g9 .* g9);
unit_energy_g10 = 3.424e-15*ones(length(g1),1) ./ (g10 .* g10);
unit_energy_g11 = 3.424e-15*ones(length(g1),1) ./ (g11 .* g11);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4, unit_energy_g5, unit_energy_g6,unit_energy_g7,...
    unit_energy_g8,unit_energy_g9,unit_energy_g10,unit_energy_g11];

for i = 1:length(output_l)-1
    if kernel_l(i) == 0 
        energy_neg3 = energy_neg3 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
            .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
            + ones(length(g1),1) .* 9.096e-12;

    else
        energy_neg3 = energy_neg3 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
            kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
            * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
            * output_w(output_l(i),1) * output_d(output_l(i),1);
    end
end

%bits_qu = log(128./q)/log(2) %how julian had it set up
q_unscaled = q./(4*SNR_qu_phase(6));
SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
bits_qu = (SNR_qu - 1.76)./6.02;

energy_q=0;
energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
num_entries = output_h(output_l(length(output_l)),1)*...
     output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
energy_q_neg3 = energy_q*num_entries;
%pad_3 = NaN(6,1);
% 
% if gvary==1
%     energy_3 = [energy_3;pad_3];
%     %size(energy_3)
%     energy_q_3=[energy_q_3;pad_3];
%     %size(energy_q_3)
%     acc5_3 = [acc5_3;pad_3];
% end

energy_tot_neg3 = energy_neg3 + energy_q_neg3;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;

%3 PHASE 

cd ../Raw
data_ga = csvread('hypercube_3P_goog_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_3P_goog_qvary_SNR.csv',1,1);
cd ../Energy

data_qu = data_qu(1:50,:);
data_ga = data_ga(1:50,:);

if gvary==1
    acc5_3= data_ga(:,1);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    g5 = data_ga(:,10);
    g6 = data_ga(:,11);
    g7 = data_ga(:,12);
    g8 = data_ga(:,13);
    g9 = data_ga(:,14);
    g10 = data_ga(:,15);
    g11 = data_ga(:,16);
    g12 = data_ga(:,17);
    g13 = data_ga(:,18);
    g14 = data_ga(:,19);
    g15 = data_ga(:,20);
    g16 = data_ga(:,21);
    g17 = data_ga(:,22);
    q = data_ga(:,23);
    
else
    acc5_3= data_qu(:,1);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    g5 = data_qu(:,10);
    g6 = data_qu(:,11);
    g7 = data_qu(:,12);
    g8 = data_qu(:,13);
    g9 = data_qu(:,14);
    g10 = data_qu(:,15);
    g11 = data_qu(:,16);
    g12 = data_qu(:,17);
    g13 = data_qu(:,18);
    g14 = data_qu(:,19);
    g15 = data_qu(:,20);
    g16 = data_qu(:,21);
    g17 = data_qu(:,22);
    q = data_qu(:,23);
end 

output_l = [2,4,5,6,7,13,14,15,16,17,19,25,26,27,...
    28,29,31,33]; %last number is for q
kernel_l = [1,0,2,3,0,4,5,6,7,8,9,10,11,12,...
    13,14,15,0];

energy = zeros(length(g1),1);
energy_3 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
unit_energy_g6 = 3.424e-15*ones(length(g1),1) ./ (g6 .* g6);
unit_energy_g7 = 3.424e-15*ones(length(g1),1) ./ (g7 .* g7);
unit_energy_g8 = 3.424e-15*ones(length(g1),1) ./ (g8 .* g8);
unit_energy_g9 = 3.424e-15*ones(length(g1),1) ./ (g9 .* g9);
unit_energy_g10 = 3.424e-15*ones(length(g1),1) ./ (g10 .* g10);
unit_energy_g11 = 3.424e-15*ones(length(g1),1) ./ (g11 .* g11);
unit_energy_g12 = 3.424e-15*ones(length(g1),1) ./ (g12 .* g12);
unit_energy_g13 = 3.424e-15*ones(length(g1),1) ./ (g13 .* g13);
unit_energy_g14 = 3.424e-15*ones(length(g1),1) ./ (g14 .* g14);
unit_energy_g15 = 3.424e-15*ones(length(g1),1) ./ (g15 .* g15);
unit_energy_g16 = 3.424e-15*ones(length(g1),1) ./ (g16 .* g16);
unit_energy_g17 = 3.424e-15*ones(length(g1),1) ./ (g17 .* g17);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4, unit_energy_g5, unit_energy_g6,unit_energy_g7,...
    unit_energy_g8,unit_energy_g9,unit_energy_g10,unit_energy_g11,...
    unit_energy_g12,unit_energy_g13,unit_energy_g14,unit_energy_g15,...
    unit_energy_g16,unit_energy_g17];

for i = 1:length(output_l)-1
    if kernel_l(i) == 0 
        energy_3 = energy_3 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
            .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
            + ones(length(g1),1) .* 9.096e-12;

    else
        energy_3 = energy_3 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
            kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
            * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
            * output_w(output_l(i),1) * output_d(output_l(i),1);
    end
end

%bits_qu = log(128./q)/log(2) %how julian had it set up
q_unscaled = q./(4*SNR_qu_phase(3));
SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
bits_qu = (SNR_qu - 1.76)./6.02;

energy_q=0;
energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
num_entries = output_h(output_l(length(output_l)),1)*...
     output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
energy_q_3 = energy_q*num_entries;
%pad_3 = NaN(6,1);
% 
% if gvary==1
%     energy_3 = [energy_3;pad_3];
%     %size(energy_3)
%     energy_q_3=[energy_q_3;pad_3];
%     %size(energy_q_3)
%     acc5_3 = [acc5_3;pad_3];
% end

energy_tot_3 = energy_3 + energy_q_3;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;

% %4 PHASE 
% 
% cd ../Raw
% data_ga = csvread('hypercube_4P_goog_gvary_SNR.csv',1,1);
% data_qu = csvread('hypercube_4P_goog_qvary_SNR.csv',1,1);
% cd ../Energy
% 
% data_qu = data_qu(1:50,:);
% data_ga = data_ga(1:50,:);
% 
% if gvary==1
%     acc5_4= data_ga(:,1);
%     g1 = data_ga(:,6);
%     g2 = data_ga(:,7);
%     g3 = data_ga(:,8);
%     g4 = data_ga(:,9);
%     g5 = data_ga(:,10);
%     g6 = data_ga(:,11);
%     g7 = data_ga(:,12);
%     g8 = data_ga(:,13);
%     g9 = data_ga(:,14);
%     g10 = data_ga(:,15);
%     g11 = data_ga(:,16);
%     g12 = data_ga(:,17);
%     g13 = data_ga(:,18);
%     g14 = data_ga(:,19);
%     g15 = data_ga(:,20);
%     g16 = data_ga(:,21);
%     g17 = data_ga(:,22);
%     g18 = data_ga(:,23);
%     g19 = data_ga(:,24);
%     g20 = data_ga(:,25);
%     g21 = data_ga(:,26);
%     g22 = data_ga(:,27);
%     g23 = data_ga(:,28);
%     q = data_ga(:,30);
% else
%     acc5_4= data_qu(:,1);
%     g1 = data_qu(:,6);
%     g2 = data_qu(:,7);
%     g3 = data_qu(:,8);
%     g4 = data_qu(:,9);
%     g5 = data_qu(:,10);
%     g6 = data_qu(:,11);
%     g7 = data_qu(:,12);
%     g8 = data_qu(:,13);
%     g9 = data_qu(:,14);
%     g10 = data_qu(:,15);
%     g11 = data_qu(:,16);
%     g12 = data_qu(:,17);
%     g13 = data_qu(:,18);
%     g14 = data_qu(:,19);
%     g15 = data_qu(:,20);
%     g16 = data_qu(:,21);
%     g17 = data_qu(:,22);
%     g18 = data_qu(:,23);
%     g19 = data_qu(:,24);
%     g20 = data_qu(:,25);
%     g21 = data_qu(:,26);
%     g22 = data_qu(:,27);
%     g23 = data_qu(:,28);
%     q = data_qu(:,29);
% end 
% 
% 
% output_l = [2,4,5,6,7,13,14,15,16,17,19,25,26,27,28,...
%     29,31,38,39,40,41,42,44,45];
% kernel_l = [1,0,2,3,0,4,5,6,7,8,9,10,11,12,13,...
%     14,15,16,17,18,19,20,21,0];
%     
% energy = zeros(length(g1),1);
% energy_4 = zeros(length(g1),1);
% unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
% unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
% unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
% unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
% unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
% unit_energy_g6 = 3.424e-15*ones(length(g1),1) ./ (g6 .* g6);
% unit_energy_g7 = 3.424e-15*ones(length(g1),1) ./ (g7 .* g7);
% unit_energy_g8 = 3.424e-15*ones(length(g1),1) ./ (g8 .* g8);
% unit_energy_g9 = 3.424e-15*ones(length(g1),1) ./ (g9 .* g9);
% unit_energy_g10 = 3.424e-15*ones(length(g1),1) ./ (g10 .* g10);
% unit_energy_g11 = 3.424e-15*ones(length(g1),1) ./ (g11 .* g11);
% unit_energy_g12 = 3.424e-15*ones(length(g1),1) ./ (g12 .* g12);
% unit_energy_g13 = 3.424e-15*ones(length(g1),1) ./ (g13 .* g13);
% unit_energy_g14 = 3.424e-15*ones(length(g1),1) ./ (g14 .* g14);
% unit_energy_g15 = 3.424e-15*ones(length(g1),1) ./ (g15 .* g15);
% unit_energy_g16 = 3.424e-15*ones(length(g1),1) ./ (g16 .* g16);
% unit_energy_g17 = 3.424e-15*ones(length(g1),1) ./ (g17 .* g17);
% unit_energy_g18 = 3.424e-15*ones(length(g1),1) ./ (g18 .* g18);
% unit_energy_g19 = 3.424e-15*ones(length(g1),1) ./ (g19 .* g19);
% unit_energy_g20 = 3.424e-15*ones(length(g1),1) ./ (g20 .* g20);
% unit_energy_g21 = 3.424e-15*ones(length(g1),1) ./ (g21 .* g21);
% unit_energy_g22 = 3.424e-15*ones(length(g1),1) ./ (g22 .* g22);
% unit_energy_g23 = 3.424e-15*ones(length(g1),1) ./ (g23 .* g23);
% 
% unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3,...
%     unit_energy_g4, unit_energy_g5, unit_energy_g6,unit_energy_g7,...
%     unit_energy_g8,unit_energy_g9,unit_energy_g10,unit_energy_g11,...
%     unit_energy_g12,unit_energy_g13,unit_energy_g14,unit_energy_g15,...
%     unit_energy_g16,unit_energy_g17,unit_energy_g18,unit_energy_g19,...
%     unit_energy_g20,unit_energy_g21,unit_energy_g22,unit_energy_g23];
% 
% for i = 1:length(output_l)-1
%     if kernel_l(i) == 0 
%         energy_4 = energy_4 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
%             .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
%             + ones(length(g1),1) .* 9.096e-12;
% 
%     else
%         energy_4 = energy_4 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
%             kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
%             * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
%             * output_w(output_l(i),1) * output_d(output_l(i),1);
%     end
% end
% 
% %bits_qu = log(128./q)/log(2) %how julian had it set up
% q_unscaled = q./(4*SNR_qu_phase(4));
% SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
% bits_qu = (SNR_qu - 1.76)./6.02;
% 
% energy_q=0;
% energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
% num_entries = output_h(output_l(length(output_l)),1)*...
%      output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
% energy_q_4 = energy_q*num_entries;
% energy_4_tot = energy_4 + energy_q_4;
% 
% g_unscaled = g1./(4*SNR_ga_phase);
% SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
% bits_ga = (SNR_ga - 1.76)./6.02;

% 
% %5 PHASE
% 
% cd ../Raw
% data_ga = csvread('hypercube_5P_goog_gvary_SNR.csv',1,1);
% data_qu = csvread('hypercube_5P_goog_qvary_SNR.csv',1,1);
% cd ../Energy
% 
% if gvary==1
%     acc5_5= data_ga(:,1);
%     g1 = data_ga(:,6);
%     g2 = data_ga(:,7);
%     g3 = data_ga(:,8);
%     g4 = data_ga(:,9);
%     g5 = data_ga(:,10);
%     g6 = data_ga(:,11);
%     g7 = data_ga(:,12);
%     g8 = data_ga(:,13);
%     g9 = data_ga(:,14);
%     g10 = data_ga(:,15);
%     g11 = data_ga(:,16);
%     g12 = data_ga(:,17);
%     g13 = data_ga(:,18);
%     g14 = data_ga(:,19);
%     g15 = data_ga(:,20);
%     g16 = data_ga(:,21);
%     g17 = data_ga(:,22);
%     g18 = data_ga(:,23);
%     g19 = data_ga(:,24);
%     g20 = data_ga(:,25);
%     g21 = data_ga(:,26);
%     g22 = data_ga(:,27);
%     g23 = data_ga(:,28);
%     g24 = data_ga(:,29);
%     g25 = data_ga(:,30);
%     g26 = data_ga(:,31);
%     g27 = data_ga(:,32);
%     g28 = data_ga(:,33);
%     g29 = data_ga(:,34);
%     q = data_ga(:,35);
% else
%     acc5_5= data_qu(:,1);
%     g1 = data_qu(:,6);
%     g2 = data_qu(:,7);
%     g3 = data_qu(:,8);
%     g4 = data_qu(:,9);
%     g5 = data_qu(:,10);
%     g6 = data_qu(:,11);
%     g7 = data_qu(:,12);
%     g8 = data_qu(:,13);
%     g9 = data_qu(:,14);
%     g10 = data_qu(:,15);
%     g11 = data_qu(:,16);
%     g12 = data_qu(:,17);
%     g13 = data_qu(:,18);
%     g14 = data_qu(:,19);
%     g15 = data_qu(:,20);
%     g16 = data_qu(:,21);
%     g17 = data_qu(:,22);
%     g18 = data_qu(:,23);
%     g19 = data_qu(:,24);
%     g20 = data_qu(:,25);
%     g21 = data_qu(:,26);
%     g22 = data_qu(:,27);
%     g23 = data_qu(:,28);
%     g24 = data_qu(:,29);
%     g25 = data_qu(:,30);
%     g26 = data_qu(:,31);
%     g27 = data_qu(:,32);
%     g28 = data_qu(:,33);
%     g29 = data_qu(:,34);
%     q = data_qu(:,35);
% end 
% 
% output_l = [2,4,5,6,7,13,14,15,16,17,19,25,26,27,28,...
%     29,31,38,39,40,41,42,44,50,51,52,53,54,56,57];
% kernel_l = [1,0,2,3,0,4,5,6,7,8,9,10,11,12,13,...
%     14,15,16,17,18,19,20,21,22,23,24,25,26,27,0];
% 
% energy = zeros(length(g1),1);
% energy_5 = zeros(length(g1),1);
% unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
% unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
% unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
% unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
% unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
% unit_energy_g6 = 3.424e-15*ones(length(g1),1) ./ (g6 .* g6);
% unit_energy_g7 = 3.424e-15*ones(length(g1),1) ./ (g7 .* g7);
% unit_energy_g8 = 3.424e-15*ones(length(g1),1) ./ (g8 .* g8);
% unit_energy_g9 = 3.424e-15*ones(length(g1),1) ./ (g9 .* g9);
% unit_energy_g10 = 3.424e-15*ones(length(g1),1) ./ (g10 .* g10);
% unit_energy_g11 = 3.424e-15*ones(length(g1),1) ./ (g11 .* g11);
% unit_energy_g12 = 3.424e-15*ones(length(g1),1) ./ (g12 .* g12);
% unit_energy_g13 = 3.424e-15*ones(length(g1),1) ./ (g13 .* g13);
% unit_energy_g14 = 3.424e-15*ones(length(g1),1) ./ (g14 .* g14);
% unit_energy_g15 = 3.424e-15*ones(length(g1),1) ./ (g15 .* g15);
% unit_energy_g16 = 3.424e-15*ones(length(g1),1) ./ (g16 .* g16);
% unit_energy_g17 = 3.424e-15*ones(length(g1),1) ./ (g17 .* g17);
% unit_energy_g18 = 3.424e-15*ones(length(g1),1) ./ (g18 .* g18);
% unit_energy_g19 = 3.424e-15*ones(length(g1),1) ./ (g19 .* g19);
% unit_energy_g20 = 3.424e-15*ones(length(g1),1) ./ (g20 .* g20);
% unit_energy_g21 = 3.424e-15*ones(length(g1),1) ./ (g21 .* g21);
% unit_energy_g22 = 3.424e-15*ones(length(g1),1) ./ (g22 .* g22);
% unit_energy_g23 = 3.424e-15*ones(length(g1),1) ./ (g23 .* g23);
% 
% unit_energy_g24 = 3.424e-15*ones(length(g1),1) ./ (g24 .* g24);
% unit_energy_g25 = 3.424e-15*ones(length(g1),1) ./ (g25 .* g25);
% unit_energy_g26 = 3.424e-15*ones(length(g1),1) ./ (g26 .* g26);
% unit_energy_g27 = 3.424e-15*ones(length(g1),1) ./ (g27 .* g27);
% unit_energy_g28 = 3.424e-15*ones(length(g1),1) ./ (g28 .* g28);
% unit_energy_g29 = 3.424e-15*ones(length(g1),1) ./ (g29 .* g29);
% unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3,...
%     unit_energy_g4, unit_energy_g5, unit_energy_g6,unit_energy_g7,...
%     unit_energy_g8,unit_energy_g9,unit_energy_g10,unit_energy_g11,...
%     unit_energy_g12,unit_energy_g13,unit_energy_g14,unit_energy_g15,...
%     unit_energy_g16,unit_energy_g17,unit_energy_g18,unit_energy_g19,...
%     unit_energy_g20,unit_energy_g21,unit_energy_g22,unit_energy_g23,...
%     unit_energy_g24,unit_energy_g25,unit_energy_g26,unit_energy_g27...
%     unit_energy_g28,unit_energy_g29];
% 
% for i = 1:length(output_l)-1
%     if kernel_l(i) == 0 
%         energy_5 = energy_5 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
%             .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
%             + ones(length(g1),1) .* 9.096e-12;
% 
%     else
%         energy_5 = energy_5 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
%             kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
%             * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
%             * output_w(output_l(i),1) * output_d(output_l(i),1);
%     end
% end
% 
% %bits_qu = log(128./q)/log(2) %how julian had it set up
% q_unscaled = q./(4*SNR_qu_phase(5));
% SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
% bits_qu = (SNR_qu - 1.76)./6.02;
% 
% energy_q=0;
% energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
% num_entries = output_h(output_l(length(output_l)),1)*...
%      output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
% energy_q_5 = energy_q*num_entries;
% energy_tot_5 = energy_5 + energy_q_5;
% 
% g_unscaled = g1./(4*SNR_ga_phase);
% SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
% bits_ga = (SNR_ga - 1.76)./6.02;


if gvary == 1
    figure()
    plot(SNR_ga,acc5_1,'r.-')
    hold on
    plot(SNR_ga,acc5_2,'g*-')
    plot(SNR_ga,acc5_neg3,'bs-')
    plot(SNR_ga,acc5_3,'mx-')
%     plot(SNR_ga,acc5_4,'cd-')
    figure()
    plot(SNR_ga,energy_1,'r.-')
    hold on
    %ylim([0,2.5e-2]);
    plot(SNR_ga,energy_2,'g*-')
    plot(SNR_ga,energy_neg3,'bs-')
    plot(SNR_ga,energy_3,'mx-')
%     plot(SNR_ga,energy_4,'cd-')
    negones = zeros(length(SNR_ga),1) - 1;
    for_gnu_goog_gvary = [SNR_ga, acc5_1, energy_1, acc5_2, energy_2,...
    acc5_neg3, energy_neg3, acc5_3, energy_3, acc5_4,energy_4, negones];
else
    figure()
    plot(SNR_qu,acc5_1,'r.-')
    hold on
    plot(SNR_qu,acc5_2,'g*-')
    xlim([10,80])
    plot(SNR_qu,acc5_neg3,'bs-')
    plot(SNR_qu,acc5_3,'mx-')
%     plot(SNR_qu,acc5_4,'cd-')
    hold off
    figure()
    plot(SNR_qu,energy_q_1,'r.-')
    hold on
%     xlim([10,80])
    plot(SNR_qu,energy_q_2,'g*-')
    plot(SNR_qu,energy_q_neg3,'bs-')
    plot(SNR_qu,energy_q_3,'mx-')
%     plot(SNR_qu,energy_q_4,'cd-')
    negones = zeros(length(SNR_qu),1) - 1;
    for_gnu_goog_qvary = [SNR_qu, acc5_1, energy_q_1, acc5_2, energy_q_2,...
    acc5_neg3, energy_q_neg3, acc5_3, energy_q_3, acc5_4, energy_q_4, negones];
end



% plotyy(SNR_qu,energy_1,SNR_qu,acc5_1)
% plot(SNR_qu,energy_1)
% hold on
% % plotyy(SNR_qu,energy_2,SNR_qu,acc5_2)
% plotyy(SNR_qu,energy_3,SNR_qu,acc5_3)
% plotyy(SNR_qu,energy_4,SNR_qu,acc5_4)
% plotyy(SNR_qu,energy_5,SNR_qu,acc5_5)
hold off
xlabel SNR_Q


