# RoBERTa fine-tuning with Distance Metric Learninng Losses

## Usage: Train RoBERTa-large/base models with the use of Distance Metric Learning Losses
Here is an example of using this package.

1. Train RoBERTa model on SoftTriple Loss and CrossEntropy Loss
```
python train_cross_validation.py --num-warmup-steps 10, --model-name roberta-large, --model-type triple-entropy --sample-size 20 --n-split 40 --dataset-name cr 
```
Below are explained parameters:

```--max-length``` the maximum length in number of tokens for the inputs to the transformer model

```--learning-rate ``` learning rate 

```--num-warmup-steps``` the number of steps for the warmup phase

```--eps``` Adam’s epsilon for numerical stability

```--model-name``` the name of the transformer model that is fine-tuned with DML losses. Right now it was only tested and validated on roberta-large and roberta-base

```--model-type``` the type of DML loss in. Right now the supported options are: ```softriple``` for the SoftTriple loss, ```supcon``` for the SupCon loss, ```triplet``` for the Triplet loss, ```proxyanchor``` for the ProxyAnchor loss, ```proxynca``` for the ProxyNCA loss, ```npairs``` for the NPairs loss, and ```baseline``` for the standard cross entropy loss

```--weight-decay``` decoupled weight decay to apply

```--la``` lambda parameter of the SoftTripleLoss.

```--gamma``` gamma parameter of the SoftTripleLoss.

```--margin``` margin parameter of the SoftTriple Loss and ProxyAnchor Loss.

```--centers``` number of centers parameter of the SoftTripleLoss.

```--beta``` parameter that controls the effect of DML loss and CrossEntropyLoss on the training process.

```--softmax-scale``` the exponent multiplier in the ProxyNCA Loss softmax expression.

```--alpha``` the scaling parameter for the ProxyAnchor Loss.

```--supcon-temp``` temperature parameter of the SupConLoss. Available only when model-type is ```supcon```

```--seed``` set the seed for the training process

```--output-dir``` the output directory where the model predictions and checkpoints will be written

```--save-steps``` the number of steps before checkpoint is saved

```--epochs``` number of epochs

```--num-training-steps``` the number of training steps to do.

```--per-device-train-batch-size``` the batch size of the training process

```--per-device-eval-batch-size``` the batch size of the evaluationg process

```--sample-size``` the number of observation sampled from the traing dataset - used for the few-shot learning scenarios

```--n-split``` the number of folds of the k-fold validation

```--dataset-name``` the name of the dataset

