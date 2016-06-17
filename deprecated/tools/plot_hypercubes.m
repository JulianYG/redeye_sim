%Will need later:
%date = ''
%g_layers =
%file_name =strcat('hypercube-',str(g_layers),'g1q-',date)

%For now:
%file_name = 'hypercube-2g1q-d1-b5-7_28_2015.csv';
file_name = 'hyperplane_save.csv'

cd .. 
cd hypercube
hypercube_data = csvread(file_name,1,1);
cd ..
cd hypercube_plots

energy = hypercube_data(:,3);
top5 = hypercube_data(:,1);
top1 = hypercube_data(:,2);
g0 = hypercube_data(:,4);
g1 = hypercube_data(:,5);
q = hypercube_data(:,6);

a = top1-min(top1);
b = max(a);
d = a./b;

e = energy-min(energy);
f = max(e);
g = e./f;

c = d./g;


%scatter3(g0,g1,q,40,c,'filled');
scatter3(g0,g1,q,40,top1,'filled');

xlabel g0
ylabel g1
zlabel q

