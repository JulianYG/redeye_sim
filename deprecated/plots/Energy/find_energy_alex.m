phases=3;
gvary=1; %one for gvary, 0 from qvary

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

switch phases
    case 1
    cd ../Raw
    data_ga = csvread('hypercube_1P_alex_gvary.csv',1,1);
    data_qu = csvread('hypercube_1P_alex_qvary.csv',1,1);
    cd ../Energy
    
    if gvary==1
        g = data_ga(:,4);
        g1 = data_ga(:,6);
        g2 = data_ga(:,7);
        q = data_ga(:,8);
    else
        g = data_ga(:,4);
        g1 = data_qu(:,6);
        g2 = data_qu(:,7);
        q = data_qu(:,8);
    end 
    
    output_l = [2,3,4];
    kernel_l = [1,0,0];
    
    energy = zeros(length(g1),1);
    unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
    unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
    unit_energy = [unit_energy_g1, unit_energy_g2];
    
    for i = 1:length(output_l)-1
        if kernel_l(i) == 0 
            energy = energy +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
                .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
                + ones(length(g1),1) .* 9.096e-12;

        else
            energy = energy + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
                kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
                * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
                * output_w(output_l(i),1) * output_d(output_l(i),1);
        end
    end
    
%   bits_qu = log(128./q)/log(2) %how julian had it set up
    q_unscaled = q./(4*SNR_qu_phase(phases));
    SNR_qu = 10*log10((.25)./((q_unscaled.^2)./3));%./log10(2);
    bits_qu = (SNR_qu - 1.76)./6.02;
    
    energy_q = 2.56e-12 * (SNR_qu_phase(phases)*4) ./ q + 4.97e-13 * bits_qu;
    num_entries = output_h(output_l(length(output_l)),1)*...
         output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
    energy_q = energy_q*num_entries;
    energy = energy + energy_q;
    
    g_unscaled = g1./(4*SNR_ga_phase);
    SNR_ga = 10*log10((.25)./(g_unscaled));%./log10(2);
    bits_ga = (SNR_ga - 1.76)./6.02;
    
    
    case 2
    cd ../Raw
    data_ga = csvread('hypercube_2P_alex_gvary.csv',1,1);
    data_qu = csvread('hypercube_2P_alex_qvary.csv',1,1);output_hs
    cd ../Energy
    
    if gvary==1
        g1 = data_ga(:,6);
        g2 = data_ga(:,7);
        g3 = data_ga(:,8);
        g4 = data_ga(:,9);
        q = data_ga(:,10);
    else
        g1 = data_qu(:,6);
        g2 = data_qu(:,7);
        g3 = data_qu(:,8);
        g4 = data_qu(:,9);
        q = data_qu(:,10);
    end 
    
    output_l = [2,3,5,6,7]; %last number is for q
    kernel_l = [1,0,2,0,0];
    
    energy = zeros(length(g1),1);
    unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
    unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
    unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
    unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
    unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
        unit_energy_g4];
    
    for i = 1:length(output_l)-1
        if kernel_l(i) == 0 
            energy = energy +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
                .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
                + ones(length(g1),1) .* 9.096e-12;

        else
            energy = energy + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
                kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
                * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
                * output_w(output_l(i),1) * output_d(output_l(i),1);
        end
    end
    
    q_unscaled = q./(4*SNR_qu_phase(phases));
    SNR_qu = 10*log10((.25)./((q_unscaled.^2)./3));%./log10(2);
    bits_qu = (SNR_qu - 1.76)./6.02;
    
    energy_q = 2.56e-12 * (SNR_qu_phase(phases)*4) ./ q + 4.97e-13 * bits_qu;
    num_entries = output_h(output_l(length(output_l)),1)*...
         output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
    energy_q = energy_q*num_entries;
    energy = energy + energy_q;
    
    g_unscaled = g1./(4*SNR_ga_phase);
    SNR_ga = 10*log10((.25)./(g_unscaled));%./log10(2);
    bits_ga = (SNR_ga - 1.76)./6.02;
        
    case 3
    cd ../Raw
    data_ga = csvread('hypercube_3P_alex_gvary.csv',1,1);
    data_qu = csvread('hypercube_3P_alex_qvary.csv',1,1);
    cd ../Energy
    
    if gvary==1
        g1 = data_ga(:,6);
        g2 = data_ga(:,7);
        g3 = data_ga(:,8);
        g4 = data_ga(:,9);
        g5 = data_ga(:,10);
        q = data_ga(:,11);
    else
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
    unit_energy_g1 = 3.424e-15*ones(length(g1),1) ./ (g1 .* g1);
    unit_energy_g2 = 3.424e-15*ones(length(g1),1) ./ (g2 .* g2);
    unit_energy_g3 = 3.424e-15*ones(length(g1),1) ./ (g3 .* g3);
    unit_energy_g4 = 3.424e-15*ones(length(g1),1) ./ (g4 .* g4);
    unit_energy_g5 = 3.424e-15*ones(length(g1),1) ./ (g5 .* g5);
    unit_energy = [unit_energy_g1, unit_energy_g2, unit_energy_g3...
        unit_energy_g4, unit_energy_g5];
    
    for i = 1:length(output_l)-1
        if kernel_l(i) == 0 
            energy = energy +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
                .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
                + ones(length(g1),1) .* 9.096e-12;

        else
            energy = energy + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
                kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
                * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
                * output_w(output_l(i),1) * output_d(output_l(i),1);
        end
    end
    
    %bits_qu = log(128./q)/log(2) %how julian had it set up
    q_unscaled = q./(4*SNR_qu_phase(phases));
    SNR_qu = 10*log10((.25)./((q_unscaled.^2)./3));%./log10(2);
    bits_qu = (SNR_qu - 1.76)./6.02;
    
    energy_q = 2.56e-12 * (SNR_qu_phase(phases)*4) ./ q + 4.97e-13 * bits_qu;
    num_entries = output_h(output_l(length(output_l)),1)*...
         output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
    energy_q = energy_q*num_entries;
    energy = energy + energy_q;
    
    g_unscaled = g1./(4*SNR_ga_phase);
    SNR_ga = 10*log10((.25)./(g_unscaled));%./log10(2);
    bits_ga = (SNR_ga - 1.76)./6.02;
    
    case 4
    cd ../Raw
    data_ga = csvread('hypercube_4P_alex_gvary.csv',1,1);
    data_qu = csvread('hypercube_4P_alex_qvary.csv',1,1);
    cd ../Energy
    
    if gvary==1
        g1 = data_ga(:,6);
        g2 = data_ga(:,7);
        g3 = data_ga(:,8);
        g4 = data_ga(:,9);
        g5 = data_ga(:,10);
        g6 = data_ga(:,11);
        q = data_ga(:,12);
    else
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
            energy = energy +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
                .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
                + ones(length(g1),1) .* 9.096e-12;

        else
            energy = energy + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
                kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
                * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
                * output_w(output_l(i),1) * output_d(output_l(i),1);
        end
    end
    
    %bits_qu = log(128./q)/log(2) %how julian had it set up
    q_unscaled = q./(4*SNR_qu_phase(phases));
    SNR_qu = 10*log10((.25)./((q_unscaled.^2)./3));%./log10(2);
    bits_qu = (SNR_qu - 1.76)./6.02;
    
    energy_q = 2.56e-12 * (SNR_qu_phase(phases)*4) ./ q + 4.97e-13 * bits_qu;
    num_entries = output_h(output_l(length(output_l)),1)*...
         output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
    energy_q = energy_q*num_entries;
    energy = energy + energy_q;
    
    g_unscaled = g1./(4*SNR_ga_phase);
    SNR_ga = 10*log10((.25)./(g_unscaled));%./log10(2);
    bits_ga = (SNR_ga - 1.76)./6.02;
   
    case 5
    cd ../Raw
    data_ga = csvread('hypercube_5P_alex_gvary.csv',1,1);
    data_qu = csvread('hypercube_5P_alex_qvary.csv',1,1);
    cd ../Energy
    
    if gvary==1
        g1 = data_ga(:,6);
        g2 = data_ga(:,7);
        g3 = data_ga(:,8);
        g4 = data_ga(:,9);
        g5 = data_ga(:,10);
        g6 = data_ga(:,11);
        g7 = data_ga(:,12);
        q = data_ga(:,13);
    else
        g1 = data_qu(:,6);
        g2 = data_qu(:,7);
        g3 = data_qu(:,8);
        g4 = data_qu(:,9);
        g5 = data_qu(:,10);
        g6 = data_qu(:,11);
        g7 = data_ga(:,12);
        q = data_qu(:,13);
    end 
    
    output_l = [2,3,5,6,8,9,10,11]; %last number is for q
    kernel_l = [1,0,2,0,3,4,5,0];
    
    energy = zeros(length(g1),1);
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
            energy = energy +  unit_energy(:,i) .* 3 .* output_h(output_l(i),1) ...
                .* output_w(output_l(i),1) .* output_d(output_l(i),1)...
                + ones(length(g1),1) .* 9.096e-12;

        else
            energy = energy + unit_energy(:,i) * (kernel_h(kernel_l(i),1) + 2) * ...
                kernel_w(kernel_l(i),1) * kernel_w(kernel_l(i),1)...
                * kernel_d(kernel_l(i),1) * output_h(output_l(i),1)...
                * output_w(output_l(i),1) * output_d(output_l(i),1);
        end
    end
    
    %bits_qu = log(128./q)/log(2) %how julian had it set up
    q_unscaled = q./(4*SNR_qu_phase(phases));
    SNR_qu = 10*log10((.25)./((q_unscaled.^2)./3));%./log10(2);
    bits_qu = (SNR_qu - 1.76)./6.02;
    
    energy_q = 2.56e-12 * (SNR_qu_phase(phases)*4) ./ q + 4.97e-13 * bits_qu;
    num_entries = output_h(output_l(length(output_l)),1)*...
         output_w(output_l(length(output_l)),1)*output_d(output_l(length(output_l)),1);
    energy_q = energy_q*num_entries;
    energy = energy + energy_q;
    
    g_unscaled = g1./(4*SNR_ga_phase);
    SNR_ga = 10*log10((.25)./(g_unscaled));%./log10(2);
    bits_ga = (SNR_ga - 1.76)./6.02;
    
end

b = num2str(phases);

if gvary == 1
    plot(SNR_ga,energy,'o')
    title(b)
else
    plot(SNR_qu,energy,'o')
    title(b)
end


