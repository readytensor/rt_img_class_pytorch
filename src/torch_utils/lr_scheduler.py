import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineAnnealing(LambdaLR):
    def __init__(self, optimizer, base_lr, warmup_epochs, num_epochs):
        lr_lambda = self.warmup_cosine_annealing(
            base_lr=base_lr, warmup_epochs=warmup_epochs, num_epochs=num_epochs
        )
        super().__init__(optimizer, lr_lambda=lr_lambda)

    @staticmethod
    def warmup_cosine_annealing(base_lr, warmup_epochs, num_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return base_lr * (epoch / warmup_epochs)
            else:
                return base_lr * (
                    0.5
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (epoch - warmup_epochs)
                            / (num_epochs - warmup_epochs)
                        )
                    )
                )

        return lr_lambda
