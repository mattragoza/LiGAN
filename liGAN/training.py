import pandas as pd
import torch
from torch import nn


class Solver(nn.Module):

    def __init__(
        self,
        train_data,
        test_data,
        model,
        loss_fn,
        optim_type,
        save_prefix,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        self.device = device

        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optim_type(model.parameters(), **kwargs)
        self.save_prefix = save_prefix

        # keep track of current training iteration
        self.curr_iter = 0

        # track running avg loss for printing
        self.total_train_loss = 0
        self.total_train_iters = 0
        self.total_test_loss = 0
        self.total_test_iters = 0

        # set up a data frame of metrics wrt training iteration
        index_cols = ['iteration', 'phase']
        self.metrics = pd.DataFrame(columns=index_cols).set_index(index_cols)

    def save_state(self):
        save_file = self.save_prefix + '_iter_' + str(self.curr_iter) + '.checkpoint'
        checkpoint = dict(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            curr_iter=self.curr_iter,
            metrics=self.metrics,
        )
        torch.save(checkpoint, save_file)
        csv_file = self.save_prefix + '.train_metrics'
        self.metrics.to_csv(csv_file, sep=' ')

    def load_state(self, save_file):
        checkpoint = torch.load(save_file)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.curr_iter = checkpoint['curr_iter']
        self.metrics = checkpoint['metrics']

    def print_metrics(self):

        train_loss = self.total_train_loss / self.total_train_iters
        self.total_train_loss = self.total_train_iters = 0
        s = '[Iteration {}] train_loss = {}'.format(
            self.curr_iter, train_loss
        )

        if self.total_test_iters > 0: # append test loss
            test_loss = self.total_test_loss / self.total_test_iters
            self.total_test_loss = self.total_test_iters = 0
            s += '\ttest_loss = {}'.format(test_loss)

        print(s)

    def forward(self, data):
        inputs, labels = data.forward()
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, labels)
        return predictions, loss

    def step(self, update=True):
        predictions, loss = self.forward(self.train_data)

        idx = (self.curr_iter, 'train')
        self.metrics.loc[idx, 'loss'] = loss.item()
        self.total_train_loss += loss.item()
        self.total_train_iters += 1

        if update:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return predictions, loss

    def test(self, n_iters):

        losses = []
        for i in range(n_iters):
            predictions, loss = self.forward(self.test_data)
            losses.append(loss.item())

        total_loss = sum(losses)
        mean_loss = total_loss / n_iters

        self.metrics.loc[(self.curr_iter, 'test'), 'loss'] = mean_loss
        self.total_test_loss += total_loss
        self.total_test_iters += n_iters

        return mean_loss

    def train(
        self,
        n_iters,
        test_interval,
        test_iters, 
        save_interval,
        print_interval,
    ):
        while self.curr_iter <= n_iters:

            if self.curr_iter % test_interval == 0:
                self.test(test_iters)

            if self.curr_iter % save_interval == 0:
                self.save_state()

            self.step(self.curr_iter < n_iters)

            if self.curr_iter % print_interval == 0:
                self.print_metrics()

            self.curr_iter += 1

        self.curr_iter = n_iters


class AESolver(Solver):

    def forward(self, data):
        inputs, _ = data.forward()
        generated, latents = self.model(inputs)
        loss = self.loss_fn(generated, inputs)
        return generated, loss


class CESolver(Solver):

    def forward(self, data):
        (context, missing), _ = data.forward()
        generated, latents = self.model(context)
        loss = self.loss_fn(generated, missing)
        return generated, loss


class VAESolver(Solver):

    def kl_div(self, latents):
        return 0

    def forward(self, data):
        inputs, _ = data.forward()
        generated, latents = self.model(inputs)
        loss = self.loss_fn(generated, inputs) + self.kl_div(latents)
        return generated, loss


class CVAESolver(VAESolver):

    def forward(self, data):
        (conditions, inputs), _ = data.forward()
        generated, latents = self.model(conditions, inputs)
        loss = self.loss_fn(generated, inputs) + self.kl_div(latents)
        return generated, loss


class GANSolver(nn.Module):

    def __init__(
        self,
        train_data,
        test_data,
        gen_model,
        disc_model,
        loss_fn,
        optim_type,
        save_prefix,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        self.device = device

        self.train_data = train_data
        self.test_data = test_data
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.loss_fn = loss_fn
        self.gen_optimizer = optim_type(gen_model.parameters(), **kwargs)
        self.disc_optimizer = optim_type(disc_model.parameters(), **kwargs)
        self.save_prefix = save_prefix
        
        # keep track of current training iteration
        self.curr_iter = 0

        # set up data frame of metrics wrt train iteration, phase, and agent
        index_cols = ['iteration', 'phase', 'agent']
        self.metrics = pd.DataFrame(columns=index_cols).set_index(index_cols)

    def disc_forward(self, data, real):
        '''
        Compute predictions and loss for the discriminator's
        ability to correctly classify real or generated data.
        '''
        with torch.no_grad(): # do not backprop to generator or data

            if real: # get real examples
                inputs, _ = data.forward()
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                latents = torch.randn(
                    data.batch_size,
                    self.gen_model.n_input,
                    device=self.device
                )
                inputs = self.gen_model(latents)
                labels = torch.zeros(data.batch_size, 1, device=self.device)

        predictions = self.disc_model(inputs)
        loss = self.loss_fn(predictions, labels)
        return predictions, loss

    def gen_forward(self, data):
        '''
        Compute predictions and loss for the generator's ability
        to produce data that is misclassified by the discriminator.
        '''
        # get generated examples
        latents = torch.randn(
            data.batch_size, self.gen_model.n_input, device=self.device
        )
        inputs = self.gen_model(latents)
        labels = torch.ones(data.batch_size, 1, device=self.device)

        predictions = self.disc_model(inputs)
        loss = self.loss_fn(predictions, labels)
        return predictions, loss

    def disc_step(self, real, update=True):

        predictions, loss = self.disc_forward(self.train_data, real)
        
        idx = (self.curr_iter, 'train', 'disc')
        self.metrics.loc[idx, 'loss'] = loss.item()

        if update:
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

        return predictions, loss

    def gen_step(self, update=True):

        predictions, loss = self.gen_forward(self.train_data)

        idx = (self.curr_iter, 'train', 'gen')
        self.metrics.loc[idx, 'loss'] = loss.item()

        if update:
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()

        return predictions, loss

    def train_disc(self, n_iters):

        for i in range(n_iters):
            real = (i%2 == 0)
            self.disc_step(real=real, update=True)

    def train_gen(self, n_iters):
        for i in range(n_iters):
            self.gen_step(update=True)
 
    def train(
        self,
        n_iters,
        n_gen_train_iters,
        n_disc_train_iters,
        test_interval,
        test_iters, 
        save_interval,
        print_interval,
    ):
        while self.curr_iter <= n_iters:

            if self.curr_iter % test_interval == 0:
                self.test(test_iters)

            if self.curr_iter % save_interval == 0:
                self.save_state()

            self.step(self.curr_iter < n_iters)

            if self.curr_iter % print_interval == 0:
                self.print_metrics()

            self.curr_iter += 1

        self.curr_iter = n_iters


class VAEGANSolver(GANSolver):
    pass


class CVAEGANSolver(GANSolver):
    pass

