### Implementation using [facenet](https://github.com/davidsandberg/facenet/wiki) (a tensorflow implementation of [FaceNet](https://arxiv.org/abs/1503.03832))
-----------------------

##### Examples:

Run without actually created clusters of dirs (which takes lots of time):

	python clustering.py  ../../data/nyt/test_aligned_60/raw2/ ../../models/facenet_models/20170512-110547/20170512-110547.pb --batch_size 100

Run the full module:

	python clustering.py  ../../data/nyt/test_aligned_60/raw2/ ../../models/facenet_models/20170512-110547/20170512-110547.pb --batch_size 100 --create_clustered_dir
