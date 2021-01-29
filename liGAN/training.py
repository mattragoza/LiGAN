import pandas as pd
import torch


class Solver(object):

    def __init__(self, data, model, loss_fn, optim_type, **kwargs):

        self.train_data = data
        self.model = model
        self.loss_fn = loss_fn

        self.optimizer = optim_type(model.parameters(), **kwargs)
        
        self.curr_iter = 0
        self.metrics = pd.DataFrame()

    def save_state(self):
        pass

    def forward(self, data):
        inputs, labels = data.forward()
        predictions = self.model(inputs)
        return predictions, self.loss_fn(predictions, labels)

    def step(self):
        predictions, loss = self.forward(self.train_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.curr_iter += 1
        return predictions, loss

    def test(self, n_iters):
        
        for i in range(n_iters):
            predictions, loss = self.forward(self.test_data)

    def train(
        self,
        n_iters,
        test_interval,
        test_iters,
        save_interval,
    ):
        for i in range(self.curr_iter, n_iters+1):

            if i % save_interval == 0:
                self.save_state()

            if i % test_interval == 0:
                self.test(test_iters)

            if i == n_iters:
                break

            self.step()
