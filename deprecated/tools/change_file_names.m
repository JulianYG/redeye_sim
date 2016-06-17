root = '/home/julian/lpirc/ILSVRC2013_DET_bbox_train';
a = dir(root);

for j = 4:length(a)
    d = a(j).name
    for i = 1:815
	    name = synsets(i).WNID;
	     
	    if strcmp(name,d)==1
		new_name = num2str([synsets(i).ILSVRC2013_DET_ID]);
		movefile(fullfile(root, name), fullfile(root, new_name));
	    end
    
    end
end

