import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN





import ast
import csv
import inspect
import logging
import os
from argparse import Namespace
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, IO, MutableMapping, Optional, Union
from warnings import warn

import torch
import yaml

from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, AttributeDict, rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.parsing import parse_class_init_keys

CHECKPOINT_HYPER_PARAMS_KEY = 'hyper_parameters'
CHECKPOINT_HYPER_PARAMS_NAME = 'hparams_name'
CHECKPOINT_HYPER_PARAMS_TYPE = 'hparams_type'

class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        print("NUM CLASSES xMOdalKD ", self.num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        if logits.dim()==1:
            print("Logits is shape of num classes %i, adding another dimension" % logits.shape[0])
            logits = torch.unsqueeze(logits, 0)
        assert logits.dim()>1, "Logits dimensions is one or less %i" %logits.dim()

        ce_loss = self.ce_loss(logits, labels)
        assert logits.dim()>1, "Logits dimensions is one or less %i" %logits.dim()
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def fusion_to_single_KD(self, data_dict, idx):
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']

        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

        # modality fusion
        feat_learner = F.relu(self.leaners[idx](pts_feat))
        feat_cat = torch.cat([img_feat, feat_learner], 1)
        feat_cat = self.fcs1[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2[idx](feat_cat))
        fuse_feat = F.relu(feat_cat * feat_weight)

        # fusion prediction
        fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

        # KL divergence
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales
        return loss, fuse_feat

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []

        for idx in range(self.num_scales):
            singlescale_loss, fuse_feat = self.fusion_to_single_KD(data_dict, idx)
            img_seg_feat.append(fuse_feat)
            loss += singlescale_loss

        try:
            img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
            loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
            # print("loss shape ", loss.shape)
            # print("data dict loss shape ",  data_dict['loss'].shape )
            data_dict['loss'] += loss
            
            return data_dict
        except Exception as e:
            print("error ", e) # error  output with shape [] doesn't match the broadcast shape [0, 25]
            return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)

        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.model_2d = ResNetFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
            self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        if self.training and not self.baseline_only:
            data_dict = self.model_2d(data_dict)
            data_dict = self.fusion(data_dict)

        return data_dict
    
    # def load_checkpoint_expand(self, checkpoint_path, strict, **kwargs):
    #     checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

    #     # for past checkpoint need to add the new key
    #     if CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
    #         checkpoint[CHECKPOINT_HYPER_PARAMS_KEY] = {}
    #     # override the hparams with values that were passed in
    #     checkpoint[CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

    #     model = self._load_model_state_extension(checkpoint, strict=strict, **kwargs)
    #     return model

    # def _load_model_state_extension(self, checkpoint, strict, **kwargs):
    #     import pdb; pdb.set_trace()
    #     return None


    # @classmethod
    # def _load_model_state(cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs_new):
    #     cls_spec = inspect.getfullargspec(cls.__init__)
    #     cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

    #     self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    #     drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
    #     cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

    #     cls_kwargs_loaded = {}
    #     # pass in the values we saved automatically
    #     if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:

    #         # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
    #         for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
    #             cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

    #         # 2. Try to restore model hparams from checkpoint using the new key
    #         _new_hparam_key = cls.CHECKPOINT_HYPER_PARAMS_KEY
    #         cls_kwargs_loaded.update(checkpoint.get(_new_hparam_key))

    #         # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
    #         cls_kwargs_loaded = _convert_loaded_hparams(
    #             cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE)
    #         )

    #         # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
    #         args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
    #         if args_name and args_name in cls_init_args_name:
    #             cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

    #     _cls_kwargs = {}
    #     _cls_kwargs.update(cls_kwargs_loaded)
    #     _cls_kwargs.update(cls_kwargs_new)

    #     if not cls_spec.varkw:
    #         # filter kwargs according to class init unless it allows any argument via kwargs
    #         _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

    #     model = cls(**_cls_kwargs)

    #     # give model a chance to load something
    #     model.on_load_checkpoint(checkpoint)

    #     # load the state_dict on the model automatically
    #     model.load_state_dict(checkpoint['state_dict'], strict=strict)

    #     return model
