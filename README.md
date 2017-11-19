# Greedy Gerbil (Working title)
## Usage

### Create one-hot vectors from data sets

To execute this step, make sure the following files are located inside the
`/data/` folder:

* `vqa_annotations_train.gzip`
* `vqa_annotations_valid.gzip`
* `vqa_annotations_test.gzip`
* `vqa_questions_train.gzip`
* `vqa_questions_valid.gzip`
* `vqa_questions_test.gzip`

These files contain questions and answers concerning the images as well
as some additional information, divided into three splits for training,
validation and testing.

To convert this data into one-hot vectors, execute the `one-hot` module:

    python3 one_hot.py

More information about how this module works can be found in the documentation
added inside the module.

After a successful execution of the module, the `/data/` folder should now
also contain the following files:

* `qa_vocab.pickle`
* `vqa_vecs_train.pickle`
* `vqa_vecs_valid.pickle`
* `vqa_vecs_test.pickle`

### Load a data set with one-hot vectors and image features

For the next step, place the following files containing the image features
and the resolution of their indices into `/data/`:

* `VQA_image_features.h5`
* `VQA_img_features2id.json`

Now, in your python code, you can use the `VQADataset` class:

```python
from data_loading import VQADataset
vec_collection = VQADataset(
    load_path="./data/vqa_vecs_train.pickle",
    image_features_path="./data/VQA_image_features.h5",
    image_features2id_path="./data/VQA_img_features2id.json"
)

dataset_loader = DataLoader(vec_collection, batch_size=4, shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataset_loader):
    print(i_batch, sample_batched)
```

Using the arguments `image_features_path` and `image_features2id_path` for
`VQADataset` is optional, but omitting these will **not** load the image features
to the data set.


### TODO
See [this GitHUb issue](https://github.com/Kaleidophon/greedy-gerbil/issues/4)
describing the project's milestones.

