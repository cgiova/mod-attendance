import math
from tensorflow import keras

def get_lr_callback(batch_size=8, mode="cos", epochs=10, plot=False):
    lr_start = 2.5e-5
    lr_max   = 5e-6 * batch_size
    lr_min   = 0.8e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    # Guard: ensure ramp makes sense
    if lr_max < lr_start:
        lr_max = lr_start

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == "exp":
            lr = (lr_max - lr_min) * (lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep)) + lr_min
        elif mode == "step":
            lr = lr_max * (lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2))
        elif mode == "cos":
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep
            decay_epoch_index  = epoch - lr_ramp_ep - lr_sus_ep
            # avoid div by zero
            decay_total_epochs = max(1, decay_total_epochs)
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return float(lr)

    if plot:
        import numpy as np
        import matplotlib.pyplot as plt
        xs = np.arange(epochs)
        ys = [lrfn(int(e)) for e in xs]
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("epoch"); plt.ylabel("lr"); plt.title(f"LR Scheduler ({mode})")
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=0)