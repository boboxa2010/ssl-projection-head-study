Bellow all experiments from the original paper are written with how it should like using our code

1. SL experiment: Resnet classic pretrain with projection MLP head on Cifar100 on super classes prediciton:

``` model = ProjectedClassifier(n_classes = 20, encoder = 'ResNet32', mode = 'mlp') ```
``` criterion = CrossEntropyLoss() ```

```
Optimizer: SGD
LR: 0.1
Momentum: 0.9
Weight Decay: 5e-4
Batch Size: 128
Epochs: 200 
```

2. SL experiment: Resnet classic pretrain with projection fixed reweightening head on Cifar100 on super classes prediciton with kappa = 1.2:

``` model = ProjectedClassifier(n_classes = 20, encoder = 'ResNet32', mode = 'fixed', kappa = 1.2) ```
``` criterion = CrossEntropyLoss() ```

```
Optimizer: SGD
LR: 0.1
Momentum: 0.9
Weight Decay: 5e-4
Batch Size: 128
Epochs: 200
Kappa: 1.2 
```

3. SCL experiment: Resnet pretrain on Cifar100 super classes with contrastive SCL loss and MLP proj head:

``` model = ProjectedClassifier(encoder = 'ResNet32', mode = 'mlp') ```
``` criterion = SupConLoss(temperature = 0.5) ```

```
Optimizer: SGD
LR: 0.1
Momentum: 0.9
Weight Decay: 1e-6
Batch Size: 512
Epochs: 400 
```

4. SCL experiment: Resnet pretrain on Cifar100 suoer classes with contrastive SCL loss and fixed reweightening proj head with kappa = 1.5:

``` model = ProjectedClassifier(encoder = 'ResNet32', mode = 'fixed', kappa = 1.5) ```
``` criterion = SupConLoss(temperature = 0.5) ```

```
Optimizer: SGD
LR: 0.1
Momentum: 0.9
Weight Decay: 1e-6
Batch Size: 512
Epochs: 400
Kappa: 1.5 
```

5.  SSL experiment: Resnet pretrain on Mnist-on-Cifar10 with SimCLR loss and MLP proj head:

``` model = SimCLR(encoder = 'ResNet32', mode = 'mlp') ```
``` criterion = SimCLRLoss(temperature=0.5) ```

```
Optimizer: Adam
LR: 0.001 (1e-3)
Weight Decay: 1e-6
Batch Size: 512
Epochs: 400
Temperature: 0.5 
```

6. SSL experiment: Resnet pretrain on Mnist-on-Cifar10 with SimCLR loss and fixed reweightening proj head with kappa = 1.05:

``` model = SimCLR(encoder = 'ResNet32', mode = 'fixed', kappa = 1.05) ```
``` criterion = SimCLRLoss(temperature=0.5) ```

```
Optimizer: Adam
LR: 0.001 (1e-3)
Weight Decay: 1e-6
Batch Size: 512
Epochs: 400
Temperature: 0.5
Kappa: 1.05 
```

In experiments above all classes return features (outputs before projection head) and projections (outputs after projection head) - need to evaluate linear probing with features and projections as outputs for results like in the paper.