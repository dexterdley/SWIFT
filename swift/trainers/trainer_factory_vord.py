# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from swift.utils import get_logger

logger = get_logger()


class TrainerFactoryVORD:
    TRAINER_MAPPING = {
        'causal_lm': 'swift.trainers.trainers_vord.Seq2SeqTrainerVORD',
        'seq_cls': 'swift.trainers.trainers_vord.TrainerVORD',
        'dpo': 'swift.trainers.trainers_vord.DPOTrainer',
        'orpo': 'swift.trainers.trainers_vord.ORPOTrainer',
        'kto': 'swift.trainers.trainers_vord.KTOTrainer',
        'cpo': 'swift.trainers.trainers_vord.CPOTrainer',
        'rm': 'swift.trainers.trainers_vord.RewardTrainer',
        'ppo': 'swift.trainers.trainers_vord.PPOTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'swift.trainers.trainers_vord.Seq2SeqTrainingArguments',
        'seq_cls': 'swift.trainers.trainers_vord.TrainingArguments',
        'dpo': 'swift.trainers.trainers_vord.DPOConfig',
        'orpo': 'swift.trainers.trainers_vord.ORPOConfig',
        'kto': 'swift.trainers.trainers_vord.KTOConfig',
        'cpo': 'swift.trainers.trainers_vord.CPOConfig',
        'rm': 'swift.trainers.trainers_vord.RewardConfig',
        'ppo': 'swift.trainers.trainers_vord.PPOConfig',
    }

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if hasattr(args, 'rlhf_type'):
            train_method = args.rlhf_type
        else:
            train_method = args.task_type
        module_path, class_name = mapping[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args, cls.TRAINING_ARGS_MAPPING)
        args_dict = asdict(args)
        parameters = inspect.signature(training_args_cls).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        if 'ppo' in training_args_cls.__name__.lower():
            args_dict['world_size'] = args.global_world_size

        return training_args_cls(**args_dict)
