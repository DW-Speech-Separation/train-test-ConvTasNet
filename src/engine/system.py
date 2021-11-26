import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
import torch.nn.functional as F
from torch.autograd import Variable


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class System(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Note that by default, any PyTorch-Lightning hooks are *not* passed to the model.
    If you want to use Lightning hooks, add the hooks to a subclass::

        class MySystem(System):
            def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
                return self.model.on_train_batch_start(batch, batch_idx, dataloader_idx)

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.

    .. note:: By default, ``training_step`` (used by ``pytorch-lightning`` in the
        training loop) and ``validation_step`` (used for the validation loop)
        share ``common_step``. If you want different behavior for the training
        loop and the validation loop, overwrite both ``training_step`` and
        ``validation_step`` instead.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "Valid_original_loss"

    def __init__(
        self,
        model,
        optimizer,
        optimizer_cosine_similarity,
        batch_iteration,
        num_layers,
        speech_embedding,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.num_layers = num_layers,
        self.model = model
        self.batch_iteration = batch_iteration,
        self.optimizer = optimizer
        self.optimizer_cosine_similarity = optimizer_cosine_similarity,
        self.speech_embedding = speech_embedding,
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Save lightning's AttributeDict under self.hparams
        self.save_hyperparameters(self.config_to_hparams(self.config))


    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, state='Train'):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        #inputs, targets = batch
        #est_targets = self(inputs)
        #loss = self.loss_func(est_targets, targets)


        #opts,epoch_schedulers = self.optimizers()
        opt1,opt2 = self.optimizers()

        #opt1 = opts[0]
        #opt2 = opts[1]

        #print("TYPE",type(self.epoch_schedulers), self.epoch_schedulers)#len(self.epoch_schedulers))

        def closure():
            inputs, targets = batch
            est_targets = self(inputs)
            loss = self.loss_func(est_targets, targets)
            self.log(state+"_original_loss", loss,on_epoch=True, prog_bar=True)
            opt1.zero_grad()
            self.manual_backward(loss)
            return loss

        self.optimizer_step(optimizer=opt1,batch_idx=batch_nb,optimizer_idx=0, optimizer_closure=closure)

        def closure_similarity():
            inputs, targets = batch
            est_targets = self(inputs)
            loss_2 = self.loss_similarity(est_targets)
            self.log(state+"_similarity_loss", loss_2, on_epoch=True, prog_bar=True)
            opt2.zero_grad()
            self.manual_backward(loss_2)
            return loss_2

        self.optimizer_step(optimizer=opt2,batch_idx=batch_nb,optimizer_idx=1, optimizer_closure=closure_similarity)

        #self.epoch_schedulers["scheduler"].step('val_loss')
        #return loss

    def calculate_similarity(self,model, est_targets,num_layers):      
          waveform_1 = est_targets[:,0]
          waveform_2 = est_targets[:,1]
          features_1, _ = model.extract_features(waveform_1,num_layers=num_layers)
          features_2, _ = model.extract_features(waveform_2,num_layers=num_layers)        
                  
          # features_1/2 => torch.Size([3, 749, 768])  [bach_size, frames, embedding_size]
          # distance => torch.Size([3, 749]) [bach_size, frames]
          distance =  F.cosine_similarity(features_1[num_layers-1], features_2[num_layers-1], dim=2) 

          # Luego hacemos el mean por muestra y luego el mean de todas las muestras.
          distance = torch.mean(distance)#.cuda()

          distance = Variable(distance, requires_grad=True)
        

          return distance

    
    
    def loss_similarity(self,est_targets):
        similitude = self.calculate_similarity(self.speech_embedding[0],est_targets,self.num_layers[0])
        similitude_value = -1*torch.log((1-similitude)/2)

        similitude_value = Variable(similitude_value, requires_grad=True)

        return similitude_value#.cuda()


    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            torch.Tensor, the value of the loss.
        """
        #loss = self.common_step(batch, batch_nb, train=True)
        self.common_step(batch, batch_nb, state='Train')
        #self.log("loss", loss, logger=True)
       
        #return loss

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
        """
        #loss = self.common_step(batch, batch_nb, train=False)
        #self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.common_step(batch, batch_nb, state='Valid')
        

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        hp_metric = self.trainer.callback_metrics.get("Valid_original_loss", None)
        if hp_metric is not None:
            self.trainer.logger.log_metrics({"hp_metric": hp_metric}, step=self.trainer.global_step)


    def optimizer_step(self,batch_idx,optimizer,optimizer_idx,optimizer_closure,on_tpu=False,using_native_amp=False,using_lbfgs=False):            # update generator every step
            # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
            """
            Search ==> # update discriminator opt every 2 steps

            """
            if optimizer_idx == 0: # Update every step
                optimizer.step(closure=optimizer_closure)

            if optimizer_idx == 1: 
                if (batch_idx + 1) % self.batch_iteration[0] == 0: # Update every  batch_iteration step steps
                    # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                    optimizer.step(closure=optimizer_closure)
                else:
                    optimizer_closure()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)

        self.epoch_schedulers = epoch_schedulers[0]

        optimizers = [self.optimizer,self.optimizer_cosine_similarity[0]]
        return optimizers#, epoch_schedulers ,
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
