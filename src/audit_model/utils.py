from typing import Callable
import logging
import math

from easydict import EasyDict
from transformers import Trainer, AutoModelForCausalLM
import wandb


def setup_model(
    config: EasyDict, 
) -> Callable:
    """
    Return a callable for model_init parameter
    This function needs to be extended for supporting different audit models.
    """
    return lambda: AutoModelForCausalLM.from_pretrained(config.pretrained_model_name_or_path)


def DP_training():
    raise NotImplementedError


def regular_training(config, training_args, train_dataset, validation_dataset, train_helper):

    # loading the config, either from target or helper
    audit_config = config.audit.target
    if train_helper:
        audit_config = config.audit.helper

    audit_type = 'helper' if train_helper else f'target_epoch_{audit_config.optimization.epoch}'
    
    # initializing wandb for visualization
    wandb.init(
            project=config.base.project_name,
            group=audit_type,
            name=audit_config.run_name,
            config=training_args
        )

    logging.info(f"Fine-tuning the {audit_type} without DP with hyperparameters:\n{training_args}")

    trainer = Trainer(
        model_init=setup_model(audit_config),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # fine-tune the audit model
    trainer.train()

    # in target model mode, we only care about the model at the end of training. Training args should be set to not save during training.
    # in helper model mode, we  care about the best model on the validation set. Training args should be set to save the best one durinng training.
    if not train_helper:
        trainer.save_model()
    
    # Evaluating the audit model
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results:\n{eval_results}")
    logging.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    return trainer.model