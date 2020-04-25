# Training

We trained face frame classification models as well as face frame sequence classifiers. All models were trained on four of the first level splits and validated on one of the first level splits. Models that had different validation folds were ensembled for the second level training.

## Stuff we tried that worked

- Mixup
- Labelsmoothing
- Having the real and fake pairs in the same training batch
- Downscale and jpeg compression augmentation
- Extending face bounding box by a small margin before cropping and larger face size (299x299)
- Not choosing fake frames that were structurally similar to their corresponding real frame

## Stuff we tried that gave only a minor boost

- LSTM, GRU and Conv1D face sequence models

## Stuff we tried that didn't work

- Training models from scratch without imagenet pretrained weights. The differences between real and fake are probably too subtle to learn from scratch.
- [Contrastive Learning](https://arxiv.org/abs/2002.05709) didn't work with this data.
- [Power spectrum classification](https://arxiv.org/pdf/1911.00686.pdf). This works only with raw video quality.
- Audio spectrum classification. There were so few audio fakes that this didn't give any boost.