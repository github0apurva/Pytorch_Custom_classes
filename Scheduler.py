import pandas as pd
import math
import matplotlib.pyplot as plt

# visualize how the LR is getting scheduled along the overall epoch
def lr_scheduler_visualize ( epochs : int , epoch_1cycle : int , decay_rate: float, lr_max : float , lr_min : float):
    # epochs       : total number of iteration the data will see
    # epoch_1cycle : total  number of epochs by when the lr will run from max to min for 1 cos cycle (cos 0 to cos 90 )
    # decay_rate   : factor by which the max lr will decrease from 1 cycle to other
    # lr_max       : maximum lr, value where lr start from
    # lr_min       : minimum lr, value where lr ends to    
    all_lr = [0]*epochs
    for epoch in range (epochs):
        all_lr[epoch] = CosineAnnealingScheduler( epoch , epoch_1cycle , decay_rate, lr_max, lr_min )
    plt.plot(all_lr)

    
# The scheduler itself
def  CosineAnnealingScheduler( epoch : int , epoch_1cycle : int , decay_rate: float, lr_max : float , lr_min : float ) :
    # epoch        : epoch number of all epochs
    # epoch_1cycle : total  number of epochs by when the lr will run from max to min for 1 cos cycle (cos 0 to cos 90 )
    # decay_rate   : factor by which the max lr will decrease from 1 cycle to other
    # lr_max       : maximum lr, value where lr start from
    # lr_min       : minimum lr, value where lr ends to 
    lr = lr_min +  decay_rate** ( int((epoch+1+epoch_1cycle)/epoch_1cycle/2) ) * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / epoch_1cycle)) / 2
    return lr
