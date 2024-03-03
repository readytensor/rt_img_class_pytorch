class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, trace_func: callable = print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, loss: float) -> bool:
        """
        Call method to evaluate the early stopping condition.

        This method updates the early stopping logic on each call by comparing the current loss with the best loss observed. If the current loss has not improved significantly (defined by `delta`) for a number of epochs specified by `patience`, it triggers early stopping.

        Parameters:
        - loss (float): The current epoch's loss value.

        Returns:
        - bool: True if early stopping is triggered (i.e., the training should stop); False otherwise.

        Side Effects:
        - Updates the internal state, including the best score observed so far, the counter for epochs without improvement, and the early stopping flag.
        - If early stopping is triggered, prints a message indicating that training will stop.

        Note:
        - The method expects loss values where lower is better. For metrics where higher is better, you should pass the negative of the metric.
        """
        score = -loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func(f"Early stopping after {self.counter} epochs")
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
