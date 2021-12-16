# This Repo contains few useful Loss function and schedulers in Pytorch



# How to use Learning Rate Scheduler for cosine annealing with warm restarts? 

![LR](https://dataiku.gbx.novartis.net/dip/api/projects/wikis/get-uploaded-file/LR.png?projectKey=DS_AI_INNOV_CODES&uploadId=iuuJSfSR9Qj7)

## How to see the learning rate across all epoch
<div class="alert">lr_scheduler_visualize ( epochs : int , epoch_1cycle : int , decay_rate: float, lr_max : float , lr_min : float)</div>

| Parameters         |      |
| ------------ |------------ |
| ``` epochs ```      |  total number of iteration the data will see       |
| ```epoch_1cycle```      | total  number of epochs by when the lr will run from max to min    |  
| ```decay_rate```        |  factor by which the max lr will decrease from 1 cycle to other   | 
| ```lr_max```       | maximum lr    |  
| ```lr_min```       | minimum lr   |


## Using the Scheduler 
<div class="alert">CosineAnnealingScheduler( epoch : int , epoch_1cycle : int , decay_rate: float, lr_max : float , lr_min : float ) </div>

| Parameters         |      |
| ------------ |------------ |
| ``` epoch ```      |  Current epoch number of total epochs       |
| ```epoch_1cycle```      | total  number of epochs by when the lr will run from max to min    |  
| ```decay_rate```        |  factor by which the max lr will decrease from 1 cycle to other   | 
| ```lr_max```       | maximum lr    |  
| ```lr_min```       | minimum lr   |

<marquee direction="right">&lt;&gt;&lt;&nbsp;&hellip;</marquee>
<marquee direction="right">&lt;&gt;&lt;&nbsp;&hellip;</marquee>
<marquee direction="right">&lt;&gt;&lt;&nbsp;&hellip;</marquee>





# Appendix: How to use above? 
The section will provide an example usage

## 1. Import basic packages 
```
#built-in functions
import pandas as pd
import math
import matplotlib.pyplot as plt
from LR_Scheduler import *
```

## 2. Visualizing the LR   
```
lr_scheduler_visualize ( 20, 4, 0.4, 0.1 , 0.001  )
```

## 3. Define the scheduler
```
n_epoch = 20 
mobile.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = keras.optimizers.SGD(lr = 0.01),
              metrics=["accuracy"] )
lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(                  lambda epoch: CosineAnnealingScheduler ( epoch , 4, 0.4, 0.1 , 0.001  ) )
model.fit(train_data, train_y , epochs = n_epoch , 
                validation_data = (valid_data , valid_y),
                callbacks = [  lr_scheduler_cb ] )
```

## Reference: 
 _ keras CosineDecayRestarts_ : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts
  _Kaggle Page_ : https://www.kaggle.com/residentmario/cosine-annealed-warm-restart-learning-schedulers
  
  
