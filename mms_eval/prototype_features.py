"""Pinned official DINOv2-S/14 features; no target-domain classifier or labels."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

import numpy as np

from .images import load_rgb, validate_policy
from .utils import sha256_file


DINO_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_WEIGHT_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
DINO_WEIGHT_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
DINO_PREPROCESSING = {'resize_short_side':256,'center_crop':224,'interpolation':'PIL_BICUBIC',
                      'scale':'uint8_to_float32_divide_255','mean':[.485,.456,.406],
                      'std':[.229,.224,.225],'feature':'normalized_CLS_token_384'}


class DinoExtractor:
    def __init__(self, repository, weights, *, device='cpu', batch_size=32, image_policy=None):
        import torch
        import torchvision
        import PIL
        root = Path(repository).resolve()
        commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if commit != DINO_COMMIT:
            raise ValueError('Use the registered official DINOv2 source commit')
        if subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD','--']).returncode:
            raise ValueError('DINOv2 repository has modified tracked files')
        untracked = subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard'],text=True).splitlines()
        if any(p.endswith('.py') for p in untracked):
            raise ValueError('DINOv2 repository contains untracked Python source')
        if sha256_file(weights) != DINO_WEIGHT_SHA256:
            raise ValueError('Use the registered official DINOv2 ViT-S/14 backbone weights')
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:
            raise ValueError('batch_size must be positive')
        # Freeze the same ordinary PyTorch attention implementation across hosts.
        os.environ['XFORMERS_DISABLED']='1'
        if 'dinov2' in sys.modules:
            loaded = Path(sys.modules['dinov2'].__file__).resolve()
            if root not in loaded.parents:
                raise ValueError('A different DINOv2 package is already loaded')
        sys.path.insert(0,str(root))
        from dinov2.hub.backbones import dinov2_vits14
        # This official file only needs torchvision. Loading it directly avoids
        # importing unrelated training/data-loader dependencies in data/__init__.
        module_spec=importlib.util.spec_from_file_location('mms_official_dino_transforms',root/'dinov2/data/transforms.py')
        transforms=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(transforms)
        self.model = dinov2_vits14(pretrained=False)
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.transform = transforms.make_classification_eval_transform()
        self.device,self.batch_size,self.policy = device,batch_size,validate_policy(image_policy)
        self.metadata = {'backend':'official-DINOv2-ViT-S14','repository':'https://github.com/facebookresearch/dinov2',
                         'source_commit':commit,'weights_sha256':DINO_WEIGHT_SHA256,'weights_url':DINO_WEIGHT_URL,
                         'preprocessing':DINO_PREPROCESSING,'image_policy':self.policy,'device':device,
                         'batch_size':batch_size,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
                         'numpy':str(np.__version__),'Pillow':str(PIL.__version__),
                         'target_domain_training_labels_used':False,'external_pretraining':'LVD-142M; see official model card',
                         'external_pretraining_overlap_status':'not_independently_verifiable_from_available_image_ids',
                         'attention':'PyTorch with xFormers disabled','precision':'float32_no_autocast',
                         'implementation_sha256':sha256_file(__file__)}

    def extract(self, records):
        import torch
        if any('label' in r or 'class_name' in r or 'human_target_class' in r for r in records):
            raise ValueError('Strip target labels before prototype feature extraction')
        blocks=[]
        with torch.inference_mode():
            for lo in range(0,len(records),self.batch_size):
                batch=torch.stack([self.transform(load_rgb(r['path'],self.policy)) for r in records[lo:lo+self.batch_size]])
                blocks.append(self.model(batch.to(self.device)).cpu().numpy())
        values=np.concatenate(blocks)
        if values.shape != (len(records),384) or not np.isfinite(values).all():
            raise ValueError('Invalid official DINOv2 feature output')
        return {'embeddings':values,'metadata':self.metadata}
