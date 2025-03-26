# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
 
import torch
import numpy as np
import torch.nn.functional as F
import pdb
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from transformers.trainer import _is_peft_model
from swift.utils import JsonlWriter, Serializer
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin_vord import SwiftMixinVORD
from .torchacc_mixin import TorchAccMixin

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

def mixup_process(x, y, mixup_alpha=1.0):
    B = x.shape[0]
    N = len(x.shape)
    indices = torch.randperm(B).to(x.device)

    while torch.any(indices == torch.arange(B).to(x.device)):
        indices = torch.randperm(B).to(x.device) # disallow repeats, i.e., x[0] and x[0] again

    if mixup_alpha == 1.0:
        lam_shape = (B,) + (1,) * (N-1)
        lam = 0.5 * torch.distributions.Beta(mixup_alpha, mixup_alpha).sample(lam_shape).to(x.device)
        lam += 0.25
    else:
        lam = 1.0
    x_mix = x * lam + x[indices] * (1 - lam)
    return x_mix, lam

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)
    return image_tensor_cd

class TrainerVORD(SwiftMixinVORD, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None:
            loss /= self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

class Seq2SeqTrainerVORD(TorchAccMixin, SwiftMixinVORD, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_mode = self.template.mode
        self.template.set_mode('pt')
        is_multimodal = self.model.model_meta.is_multimodal
        origin_data_collator = self.data_collator

        if is_multimodal:
            self.template.remove_post_encode_hook()
        self.data_collator = self._predict_data_collator
        try:
            yield
        finally:
            if is_multimodal:
                self.template.register_post_encode_hook([self.model])
            self.data_collator = origin_data_collator
            self.template.set_mode(origin_mode)
 
    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            return super().evaluate(*args, **kwargs)
 
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        resp_list = self.infer_engine.infer(
            data_list,
            RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
            use_tqdm=False,
            template=self.template)
 
        response_list = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            self.jsonl_writer.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_kwargs = {}
        BS = self.args.per_device_train_batch_size

        labels = None
        if (self.label_smoother is not None or self.compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale

        outputs = model(**inputs)
        
        cd_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                cd_inputs[key] = value.clone()
            else:
                cd_inputs[key] = value

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        """ Perform VORD Modifications """
        unwrapped_model = self.accelerator.unwrap_model(model)
        if "deepseek-vl" in self.args.logging_dir or "InternVL2_5" in self.args.logging_dir:
            if _is_peft_model(unwrapped_model):
                ViT = unwrapped_model.base_model.model.vision_model   # deepseek = vision_model, QWEN = visual
            else:
                ViT = unwrapped_model.vision_model

        elif "llava-v1.6" in self.args.logging_dir:
            if _is_peft_model(unwrapped_model):
                ViT = unwrapped_model.base_model.model.vision_tower
            else:
                ViT = unwrapped_model.vision_tower

        elif "gemma" in self.args.logging_dir:
            if _is_peft_model(unwrapped_model):
                ViT = unwrapped_model.base_model.model.vision_tower
            else:
                ViT = unwrapped_model.vision_tower

        # Here
        images_cd, lam = mixup_process(inputs['pixel_values'], inputs['labels'])
        images_cd = add_diffusion_noise(images_cd, noise_step=500)
        cd_inputs['pixel_values'] = images_cd.to(torch.bfloat16)

        with torch.no_grad():
            cd_outputs = model(**cd_inputs)  # forward mixed images
            clean_feats = ViT(inputs['pixel_values'].squeeze(1).to(torch.bfloat16))
            cd_feats = ViT(cd_inputs['pixel_values'].squeeze(1).to(torch.bfloat16))

            if 'last_hidden_state' in clean_feats:
                cosine_similarity = cosine_sim(clean_feats['last_hidden_state'].mean(2), cd_feats['last_hidden_state'].mean(2)).clamp(-1, 1)
                angular_similarity_margin = torch.acos(cosine_similarity) / torch.tensor(np.pi).to(cosine_similarity.device)

            elif 'pooler_output' in clean_feats:
                cosine_similarity = cosine_sim(clean_feats['pooler_output'], cd_feats['pooler_output']).clamp(-1, 1)
                angular_similarity_margin = torch.acos(cosine_similarity) / torch.tensor(np.pi).to(cosine_similarity.device)

            else:
                cosine_similarity = cosine_sim(torch.vstack(clean_feats).mean(1), torch.vstack(cd_feats).mean(1)).clamp(-1, 1)
                angular_similarity_margin = torch.acos(cosine_similarity) / torch.tensor(np.pi).to(cosine_similarity.device)

            if "deepseek-vl" in self.args.logging_dir and len(angular_similarity_margin) > BS: # For deepseek high and low heads
                angular_similarity_margin = (angular_similarity_margin[:BS] + angular_similarity_margin[BS:]) / 2
                # angular_similarity_margin = angular_similarity_margin[:BS]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * (labels[:, 1:] != -100).sum() / num_items_in_batch

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

            # Here
            logits, cd_logits = outputs['logits'], cd_outputs['logits']
            probs, cd_probs = F.softmax(logits, 2), F.softmax(cd_logits, 2)
            mask = torch.where(inputs['labels'] != -100)

            max_indices = torch.argmax(probs, dim=2, keepdim=True)  # Find argmax along vocab
            max_cd_probs = torch.gather(cd_probs, dim=2, index=max_indices).squeeze(-1)

            hard_neg_mask = torch.ones_like(probs, dtype=torch.bool)
            hard_neg_mask.scatter_(2, max_indices, False)  # Mask out y' = y

            if self.args.sim_margin:  # Max vocab, L1 or L2 variant with margins
                vord_loss_a = F.relu(max_cd_probs - probs.max(2).values + angular_similarity_margin.unsqueeze(1)).pow(self.args.power) # 1st term: max(P(y|v̂, x) - P(y|v, x) + m, 0)^ψ
                vord_loss_b = F.relu( (probs - cd_probs) * hard_neg_mask + angular_similarity_margin.view(-1, 1, 1)/probs.size(2) ).pow(self.args.power).mean(2) # 2nd term: sum over y'≠y of max(P(y'|v, x) - P(y'|v̂, x) + m, 0)^ψ
            else:
                vord_loss_a = F.relu(max_cd_probs - probs.max(2).values).pow(self.args.power)
                vord_loss_b = F.relu( (probs - cd_probs) * hard_neg_mask).pow(self.args.power).mean(2)

            vord_loss = vord_loss_a + vord_loss_b
            vord_loss = vord_loss[mask].mean()
            vord_out_a = F.relu(max_cd_probs - probs.max(2).values)[mask].mean()
            vord_out_b = F.relu( (probs - cd_probs) * hard_neg_mask + angular_similarity_margin.view(-1, 1, 1)/probs.size(2) ).mean(2)[mask].mean()

            self.state.xent_loss = loss
            self.state.vord_loss_a = vord_out_a
            self.state.vord_loss_b = vord_out_b
            self.state.margin = angular_similarity_margin.unsqueeze(1).mean()

            if self.args.power > 0:
                 loss += vord_loss

        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            loss = reduce_xtuner_sequence_parallel_loss(loss, labels)
 
        if getattr(self.args, 'average_tokens_across_devices', False):
            loss *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss