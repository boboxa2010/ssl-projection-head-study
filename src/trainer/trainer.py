import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Special case for linear probe - so load only backbone without hardcode of backbome (so will work for ViT and others)

        Args:
            pretrained_path (str): path to the model state dict.
        """
        if hasattr(self.model, "probe_classifier"): # если это моя проба (то есть либо трэйн либо инференс моей пробы)

            pretrained_path = str(pretrained_path)
            if hasattr(self, "logger"):  # to support both trainer and inferencer
                self.logger.info(f"Loading model weights from: {pretrained_path} ...")
            else:
                print(f"Loading model weights from: {pretrained_path} ...")
            checkpoint = torch.load(pretrained_path, self.device)

            if checkpoint.get("state_dict") is not None:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            keys = list(state_dict.keys())
            has_classifier = any(k.startswith("probe_classifier.") for k in keys)

            if has_classifier: # то есть это подгрузка чекпоинта пробы уже либо для инференеса либо для продолжения оубучения хз че там внутри
                self.model.load_state_dict(state_dict) # в точности эта проба 
            else: # то есть это загрузка модели для пробинга над ней
                self.model.model.load_state_dict(state_dict)
            return
        #иначе чтобы не ломать не пробу возвращаю просто родителксьий метод
        super()._from_pretrained(pretrained_path)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.scaler.is_enabled()):
            if self.training_mode == "supervised" or self.training_mode == "probing": # по дефлоту такой же
                outputs = self.model(**batch)
                batch.update(outputs)
            elif self.training_mode == "CL": # contrastive loss: SCL or SSL need two augs for SimCLR or SCL
                z_i = self.model(batch["aug1"])["projections"]
                z_j = self.model(batch["aug2"])["projections"]
                batch.update({"z_i": z_i, "z_j": z_j})
            else:
                print("у тебя там все ок?")
                
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            self.scaler.scale(batch["loss"]).backward()
            
            self.scaler.unscale_(self.optimizer)
            self._clip_grad_norm()

            self.scaler.step(self.optimizer)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            img = batch["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("image", img)
            aug1 = batch["aug1"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("aug1", aug1)
            aug2 = batch["aug2"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("aug2", aug2)
        else:
            img = batch["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("image", img)
