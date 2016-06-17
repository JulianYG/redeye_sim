gvary=0;

% gvary=11; %one for gvary, 0 from qvary
file_name = 'alexnet_layer_stats.csv';

SNR_qu_phase =[37.6784,29.4393,25.573,16.875,20.0219]; %quantization scale...
...for each P (1P, 2P, 3P, 4P, 5P)
SNR_ga_phase = 58.2871;
    
data = csvread(file_name,0,1);
output_d=data(:,1);
output_h=data(:,2);
output_w=data(:,3);


kernel=[11, 11, 3; 5, 5, 48; 3, 3, 256; 3, 3, 192; 3 3 192];
kernel_h=kernel(:,1);
kernel_w=kernel(:,2);
kernel_d=kernel(:,3);

%1 PHASE

cd ../Raw
data_ga = csvread('hypercube_1P_alex_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_1P_alex_qvary_SNR.csv',1,1);
data_ga = data_ga(1:50,:);
data_qu = data_qu(1:50,:);
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

output_l = [2,3,4];
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
data_ga = csvread('hypercube_2P_alex_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_2P_alex_qvary_SNR.csv',1,1);
data_ga = data_ga(1:50,:);
data_qu = data_qu(1:50,:);
cd ../Energy

if gvary==1
    acc5_2= data_ga(:,1);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    q = data_ga(:,10);
else
    acc5_2= data_ga(:,1);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    q = data_qu(:,10);
end 

output_l = [2,3,5,6,7]; %last number is for q
kernel_l = [1,0,2,0,0];

energy = zeros(length(g1),1);
energy_2 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4];

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
        

%3 PHASE 

cd ../Raw
data_ga = csvread('hypercube_3P_alex_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_3P_alex_qvary_SNR.csv',1,1);
data_ga = data_ga(1:50,:);
data_qu = data_qu(1:50,:);
cd ../Energy

if gvary==1
    acc5_3= data_ga(:,1);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    g5 = data_ga(:,10);
    q = data_ga(:,11);
else
    acc5_3= data_qu(:,1);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    g5 = data_qu(:,10);
    q = data_qu(:,11);
end 

output_l = [2,3,5,6,8,8]; %last number is for q
kernel_l = [1,0,2,0,3,3];

energy = zeros(length(g1),1);
energy_3 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4, unit_energy_g5];

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
energy_tot_3 = energy_3 + energy_q_3;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;
    

%4 PHASE 

cd ../Raw
data_ga = csvread('hypercube_4P_alex_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_4P_alex_qvary_SNR.csv',1,1);
data_ga = data_ga(1:50,:);
data_qu = data_qu(1:50,:);
cd ../Energy

if gvary==1
    acc5_4= data_ga(:,1);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    g5 = data_ga(:,10);
    g6 = data_ga(:,11);
    q = data_ga(:,12);
else
    acc5_4= data_qu(:,1);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    g5 = data_qu(:,10);
    g6 = data_qu(:,11);
    q = data_qu(:,12);
end 

output_l = [2,3,5,6,8,9,9]; %last number is for q
kernel_l = [1,0,2,0,3,4,4];

energy = zeros(length(g1),1);
energy_4 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
unit_energy_g6 = 3.424e-15*ones(length(g1),1) ./ (g6 .* g6);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4, unit_energy_g5, unit_energy_g6];

for i = 1:length(output_l)-1
    if kernel_l(i) == 0 
        energy_4 = energy_4 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
            .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
            + ones(length(g1),1) .* 9.096e-12;

    else
        energy_4 = energy_4 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
            kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
            * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
            * output_w(output_l(i),1) * output_d(output_l(i),1);
    end
end

%bits_qu = log(128./q)/log(2) %how julian had it set up
q_unscaled = q./(4*SNR_qu_phase(4));
SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
bits_qu = (SNR_qu - 1.76)./6.02;

energy_q=0;
energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
num_entries = output_h(output_l(length(output_l)),1)*...
     output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
energy_q_4 = energy_q*num_entries;
energy_4_tot = energy_4 + energy_q_4;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;
   
    
%5 PHASE

cd ../Raw
data_ga = csvread('hypercube_5P_alex_gvary_SNR.csv',1,1);
data_qu = csvread('hypercube_5P_alex_qvary_SNR.csv',1,1);
data_ga = data_ga(1:50,:);
data_qu = data_qu(1:50,:);
cd ../Energy

if gvary==1
    acc5_5= data_ga(:,1);
    g1 = data_ga(:,6);
    g2 = data_ga(:,7);
    g3 = data_ga(:,8);
    g4 = data_ga(:,9);
    g5 = data_ga(:,10);
    g6 = data_ga(:,11);
    g7 = data_ga(:,12);
    q = data_ga(:,13);
else
    acc5_5= data_qu(:,1);
    g1 = data_qu(:,6);
    g2 = data_qu(:,7);
    g3 = data_qu(:,8);
    g4 = data_qu(:,9);
    g5 = data_qu(:,10);
    g6 = data_qu(:,11);
    g7 = data_qu(:,12);
    q = data_qu(:,13);
end 

output_l = [2,3,5,6,8,9,10,11]; %last number is for q
kernel_l = [1,0,2,0,3,4,5,0];

energy = zeros(length(g1),1);
energy_5 = zeros(length(g1),1);
unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
unit_energy_g6 = 3.424e-15*ones(length(g1),1) ./ (g6 .* g6);
unit_energy_g7 = 3.424e-15*ones(length(g1),1) ./ (g7 .* g7);
unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
    unit_energy_g4, unit_energy_g5, unit_energy_g6, unit_energy_g7];

for i = 1:length(output_l)-1
    if kernel_l(i) == 0 
        energy_5 = energy_5 +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
            .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
            + ones(length(g1),1) .* 9.096e-12;

    else
        energy_5 = energy_5 + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
            kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
            * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
            * output_w(output_l(i),1) * output_d(output_l(i),1);
    end
end

%bits_qu = log(128./q)/log(2) %how julian had it set up
q_unscaled = q./(4*SNR_qu_phase(5));
SNR_qu = 10*log10((.5)./((q_unscaled.^2)./3));%./log10(2);
bits_qu = (SNR_qu - 1.76)./6.02;

energy_q=0;
energy_q = (2.56e-12 * .5)./q_unscaled + 4.97e-13 * bits_qu;
num_entries = output_h(output_l(length(output_l)),1)*...
     output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
energy_q_5 = energy_q*num_entries;
energy_tot_5 = energy_5 + energy_q_5;

g_unscaled = g1./(4*SNR_ga_phase);
SNR_ga = 10*log10((.5)./(g_unscaled).^2);%./log10(2);
bits_ga = (SNR_ga - 1.76)./6.02;
    
%GRAPH SCRIPT BELOW 

% if gvary == 1
%     figure()
%     plot(SNR_ga,acc5_1,'r.-')
%     hold on
%     plot(SNR_ga,acc5_2,'g*-')
%     plot(SNR_ga,acc5_3,'bs-')
%     plot(SNR_ga,acc5_4,'mx-')
%     plot(SNR_ga,acc5_5,'cd-')
%     figure()
%     plot(SNR_ga,energy_tot_1,'r.-')
%     hold on
%     plot(SNR_ga,energy_tot_2,'g*-')
%     plot(SNR_ga,energy_tot_3,'bs-')
%     plot(SNR_ga,energy_tot_4,'mx-')
%     plot(SNR_ga,energy_tot_5,'cd-')
% else
%     figure()
%     plot(SNR_qu,acc5_1,'r.-')
%     hold on
%     plot(SNR_qu,acc5_2,'g*-')
%     plot(SNR_qu,acc5_3,'bs-')
%     plot(SNR_qu,acc5_4,'mx-')
%     plot(SNR_qu,acc5_5,'cd-')
%     figure()
%     plot(SNR_qu,energy_tot_1,'r.-')
%     hold on
%     plot(SNR_qu,energy_tot_2,'g*-')
%     plot(SNR_qu,energy_tot_3,'bs-')
%     plot(SNR_qu,energy_tot_4,'mx-')
%     plot(SNR_qu,energy_tot_5,'cd-')
% end

if gvary == 1
    figure()
    plot(SNR_ga,acc5_1,'r.-')
    hold on
    plot(SNR_ga,acc5_2,'g*-')
    plot(SNR_ga,acc5_3,'bs-')
    plot(SNR_ga,acc5_4,'mx-')
    plot(SNR_ga,acc5_5,'cd-')
    figure()
    plot(SNR_ga,energy_1,'r.-')
    hold on
    ylim([0,5e-3]);
    plot(SNR_ga,energy_2,'g*-')
    plot(SNR_ga,energy_3,'bs-')
    plot(SNR_ga,energy_4,'mx-')
    plot(SNR_ga,energy_5,'cd-')
    for_gnu_alex_gvary = [SNR_ga, acc5_1, energy_1, acc5_2, energy_2,...
    acc5_3, energy_3, acc5_4,energy_4, acc5_5, energy_5];
else
    figure()
    plot(SNR_qu,acc5_1,'r.-')
    hold on
    xlim([10,80])
    plot(SNR_qu,acc5_2,'g*-')
    plot(SNR_qu,acc5_3,'bs-')
    plot(SNR_qu,acc5_4,'mx-')
    plot(SNR_qu,acc5_5,'cd-')
    figure()
    plot(SNR_qu,energy_q_1,'r.-')
    hold on
    xlim([10,80])
    plot(SNR_qu,energy_q_2,'g*-')
    plot(SNR_qu,energy_q_3,'bs-')
    plot(SNR_qu,energy_q_4,'mx-')
    plot(SNR_qu,energy_q_5,'cd-')
    for_gnu_alex_qvary = [SNR_qu, acc5_1, energy_q_1, acc5_2, energy_q_2,...
    acc5_3, energy_q_3, acc5_4, energy_q_4, acc5_5, energy_q_5];
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



