import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import loguru
import time
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass

class CommonTrainer:
    def __init__(
        self,
        epochs,
        epoch_offset = 1,
        device = "cpu",
        log_interval_second = 30,
        logger = loguru.logger,
    ):
        self.__device: str = device
        self.__epochs: int = epochs
        self.__epoch_offset: int = epoch_offset
        self.__log = logger.info
        self.__log_interval_second: int = log_interval_second
        self.__last_log_time = time.time()
        self.__save_model_callback: Callable[[int, nn.Module], None] = None
        self.__evaluation_callback: Callable[[int, int, Any, Any, Any], None] = None

        self.__train_dataloader: Optional[DataLoader] = None
        self.__test_dataloader: Optional[DataLoader] = None
        self.__loss_function: nn.Module = None
        self.__model: nn.Module = None
        self.__optim: nn.Module = None

    def inject_train_dataloader(self, dataloader):
        if dataloader is None:
            raise Exception("inject invalid dataloader")
        self.__train_dataloader = dataloader
        return self

    def inject_test_dataloader(self, dataloader):
        if dataloader is None:
            raise Exception("inject invalid dataloader")
        self.__test_dataloader = dataloader
        return self

    def inject_optim(self, optim):
        if optim is None:
            raise Exception("inject invalid optim")
        self.__optim = optim
        return self

    def inject_model(self, model):
        if model is None:
            raise Exception("inject invalid model")
        self.__model = model.to(self.__device)
        return self

    def inject_loss_function(self, loss_function): 
        if loss_function is None:
            raise Exception("inject invalid loss function")
        self.__loss_function = loss_function.to(self.__device)
        return self

    def inject_save_model_callback(self, callback):
        if callback is None:
            raise Exception("inject invalid callback")
        self.__save_model_callback = callback
        return self

    def inject_evaluation_callback(self, callback):
        if callback is None:
            raise Exception("inject invalid callback")
        self.__evaluation_callback = callback
        return self

    def get_device(self):
        return self.__device

    def __metric_log(self, epoch, sample, metrics):
        lst = []
        for k, v in metrics.items():
            lst.append("{}: {:.4E}".format(k, v))
        body = ", ".join(lst)
        self.__log(f"[epoch {epoch} --- sample {sample}] {body}")

    def __do_logging(self, epoch, sample, metrics):
        now = time.time()
        if now - self.__last_log_time < self.__log_interval_second:
            return
        self.__last_log_time = now
        self.__metric_log(epoch, sample, metrics)

    def train(self):
        if self.__model is None:
            raise Exception("inject model first")
        if self.__optim is None:
            raise Exception("no optim have been injected")
        if self.__train_dataloader is None:
            raise Exception("no train data loader have been injected")
        if self.__loss_function is None:
            raise Exception("no loss function have been injected")

        for epoch in range(self.__epochs):
            epoch += self.__epoch_offset
            self.__log(f"================================================[epoch {epoch}]================================================")
            self.__log("start training")
            self.__model.train()
            for i, (x, y) in enumerate(self.__train_dataloader()):
                metrics = dict()

                self.__optim.zero_grad()
                yhat = self.__model(x)
                loss = self.__loss_function(yhat, y)
                loss.backward()
                self.__optim.step()
                metrics["training_loss"] = loss

                self.__do_logging(epoch, i, metrics)

            if self.__save_model_callback is not None:
                self.__log("start saving model")
                self.__save_model_callback(epoch, self.__model)

            if self.__test_dataloader is None:
                continue

            self.__log("start evaluating model")
            with torch.no_grad():
                self.__model.eval()
                loss_arr = []
                metrics = dict()
                for i, (x, y) in enumerate(self.__test_dataloader()):
                    yhat = self.__model(x)
                    loss = self.__loss_function(yhat, y).detach()
                    loss_arr.append(loss)
                    if self.__evaluation_callback is not None:
                        self.__evaluation_callback(epoch, i, x, y, yhat)
                metrics["evaluation_loss"] = sum(loss_arr) / len(loss_arr)
                self.__metric_log(epoch, -1, metrics)
