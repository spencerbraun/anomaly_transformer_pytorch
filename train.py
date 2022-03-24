import logging
from datetime import datetime

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from model import AnomalyTransformer

logger = logging.getLogger(__name__)


def train(config, model, train_data, val_data):

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        # collate_fn=collate_fn,
        drop_last=True,
    )
    total_steps = int(len(train_dataloader) * config.train.epochs)
    warmup_steps = max(int(total_steps * config.train.warmup_ratio), 200)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.lr,
        eps=config.train.adam_epsilon,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))

    num_steps = 0
    model.train()

    for epoch in range(int(config.train.epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):

            outputs = model(batch)
            min_loss = model.min_loss(batch)
            max_loss = model.max_loss(batch)
            min_loss.backward(retain_graph=True)
            max_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            num_steps += 1

            if not config.debug:
                wandb.log({"loss": loss.item()}, step=num_steps)

        if not config.debug:
            wandb.log(output, step=num_steps)
            torch.save(model.state_dict(), config.train.pt)


@hydra.main(config_path="./conf", config_name="config")
def main(config: DictConfig) -> None:

    set_seed(config.train.state.seed)

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    logger.info(f"Using the model: {config.model.name}")

    train_data, val_data = get_data(config)
    config.data.num_class = len(set([x["labels"] for x in train_features]))
    print(f"num_class: {config.data.num_class}")

    if not config.debug:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.train.wandb.run_name}_{config.model.model}_{config.data.name}_{timestamp}"
        wandb.init(
            entity=config.train.wandb_entity,
            project=config.train.wandb_project,
            config=dict(config),
            name=run_name,
        )
        if not config.train.pt:
            config.train.pt = f"{config.train.pt}/{run_name}"

    model = AnomalyTransformer(config)
    model.to(config.device)

    train(config, model, train_data, val_data)


if __name__ == "__main__":
    main()
