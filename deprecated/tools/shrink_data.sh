DATA_DIR=$1
cd $DATA_DIR

for folder in $(ls); do
	new_name=$DATA_DIR/../ILSVRC2012_img_deg_qual_train/${folder};
	mkdir ${new_name};
	cd ${folder};
	for file in $(ls -p | grep -v / | tail -100); do
		cp $file ${new_name};
	done;
	cd ../;
done;
