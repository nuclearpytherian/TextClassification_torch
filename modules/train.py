
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import copy

class TextTrainer:

    def __init__(self, train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, early_stoper, EPOCH):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stoper = early_stoper
        self.EPOCH = EPOCH

    def train(self):
        model_history = []
        train_losses = []
        val_losses = []
        start_time = time.time()
        for i in range(self.EPOCH):

            train_loss = 0
            val_loss = 0

            self.model.train()
            for x, label in self.train_dataloader:
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*len(x) / len(self.train_dataloader)
            if self.scheduler != None:
                self.scheduler.step()

            train_losses.append(train_loss)

            self.model.eval()
            for x, label in self.val_dataloader:
                pred = self.model(x)
                loss = self.criterion(pred, label)
                val_loss += loss.item()*len(x) / len(self.val_dataloader)
            val_losses.append(val_loss)

            model_history.append(copy.deepcopy(self.model.state_dict()))

            print("Epoch {0}/{1}. Train loss {2:.4}. Val loss {3:.4}".format(i + 1, self.EPOCH, train_loss, val_loss))

            if self.early_stoper != None:
                self.early_stoper(val_loss, self.model)
                if self.early_stoper.early_stop:
                    print("Early stop is processed.")
                    break

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.model_history = model_history

        end_time = time.time()
        self.trained_time = end_time - start_time
        print("---FINISHED---")
        print("Trained time: {0} secs".format(round(self.trained_time,3)))

    def save_model(self, best_model=True):
        if not os.path.isdir('artifact'):
            os.mkdir('artifact')
        if best_model:
            best_model_idx = np.argmin(np.array(self.val_losses))
            torch.save(self.model_history[best_model_idx], os.path.join('artifact', "best_epoch_model.pt"))
        else:
            torch.save(self.model_history[len(self.val_losses)-1], os.path.join('artifact', "last_epoch_model.pt"))

    def plot_loss_graph(self):
        fig, axes = plt.subplots(2,1, figsize=(10,6))
        x = list(range(len(self.val_losses)))
        axes[0].plot(x, self.train_losses, "-o", color='blue')
        axes[0].set_title("Training loss")
        axes[0].grid()
        fig.tight_layout()
        axes[1].plot(x, self.val_losses, "-o", color='red')
        axes[1].set_title("Validating loss")
        axes[1].set_xlabel('Epochs')
        axes[1].grid()
        fig.tight_layout()
        plt.show()


