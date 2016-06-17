root = '/home/julian/caffe'

#unzip the tar files in train folder
#for f in *.tar; do tar -xvf $f -C $(PWD) + f; done

# for deg_val
cd ${root}/deg_val
for name in ${root}/deg_val/*.JPEG; do
	convert -resize 256x256\! $name $name
	mogrify -quality 50% $name
done

# for deg_train
cd ${root}/deg_train
for folder in $(ls); do
	for name in ${root}/deg_train/${folder}/*.JPEG; do
		convert -resize 256x256\! $name $name
	done
done
find . -name '*.JPEG' -execdir mogrify -quality 50% {} \;

#remove currently existing files
cd ${root}/examples/imagenet
rm -r ilsvrc12_val_lmdb
rm -r ilsvrc12_train_lmdb

cd ${root}/data/ilsvrc12
rm imagenet_mean.binaryproto

#recreate lmdb files
cd ${root}/examples/imagenet
./create_imagenet.sh
#regenerate image mean file
./make_imagenet_mean.sh
#now do the train
./train_caffenet.sh
