
data_fd = fopen('../hyperplane_20000.txt', 'r');
full_text = fscanf(data_fd, '%c');
lines = strsplit(full_text, '\n');
g0 = zeros(1, length(lines));
g1 = zeros(1, length(lines));
g2 = zeros(1, length(lines));
q = zeros(1, length(lines));
acc_top5 = zeros(1, length(lines));
loss = zeros(1, length(lines));
for idx = 1:numel(lines) - 1
   line = lines(idx);
   items = strsplit(line{1}, '\t');
   noise_param = strsplit(items{1}, '_');
   acc = items{2};
   raw_loss = items{3};
   g0(idx) = str2double(noise_param{2});
   g1(idx) = str2double(noise_param{4});
   g2(idx) = str2double(noise_param{6});
   q(idx) = str2double(noise_param{8});
   acc_both = strsplit(acc, ',');
   top5 = acc_both{1};
   loss(idx) = str2double(raw_loss);
   acc_top5(idx) = str2double(top5(2:length(top5)));
end

