import pandas as pd
import torch


class Solver(object):

    def __init__(self, data, model, loss_fn, optim_type, **kwargs):

        self.data = data
        self.model = model
        self.loss_fn = loss_fn

        self.optimizer = optim_type(model.parameters(), **kwargs)
        
        self.curr_iter = 0
        self.metrics = pd.DataFrame()

    def forward(self):
        inputs, labels = self.data.forward()
        predictions = self.model(inputs)
        return self.loss_fn(predictions, labels)

    def step(self):
        loss = self.forward()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.curr_iter += 1

    def test(self, data, n_iters):
        
        for i in range(n_iters):
            X_true, Y_true = data.forward()
            Y_pred = self.model(X_true)
            loss = loss_fn(Y_pred, Y_true)

    def train(
        self,
        train_data,
        test_data,
        n_iters,
        test_interval,
        test_iters,
        save_interval,
    ):
        for i in range(self.curr_iter, n_iters+1):

            if i % snapshot == 0:
                self.snapshot()

            if i % test_interval == 0:
                self.test(train_data, test_iter)
                self.test(test_data, test_iter)

            if i == n_iters:
                break

            self.step(train_data)
