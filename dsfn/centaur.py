"""Centaur sensor adaptation used in the revision.

Adapted from sustainable-computing/Centaur, commit
9a8d77f33e6e734b4cadbd1ca3c38a7b633e37ff (MIT).
Copyright (c) 2023 Sanju Xaviar. See ../third_party/README.md.
The DAE/classifier mathematics and parameter names are preserved.
Inputs to the DAE are the train-fitted min-max coordinates; use affine_native
after the shared standardization/noise/masking step, as in the experiment.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SHAPE = (9, 1024)
LATENT = 200
ARCHITECTURE_VERSION = 'centaur_sensor_adapter_v1_20261021'

@contextmanager
def cpu_seed(seed):
    state=torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(int(seed));yield
    finally:torch.random.set_rng_state(state)

def validate(x):
    assert x.ndim==3 and tuple(x.shape[1:])==SHAPE and len(x)>0
    assert torch.is_floating_point(x) and torch.isfinite(x).all()

def affine_native(x,minimum,maximum):
    """No clipping; zero whole modalities are detected before affine mapping."""
    validate(x)
    minimum=torch.as_tensor(minimum,dtype=x.dtype,device=x.device).view(1,9,1)
    maximum=torch.as_tensor(maximum,dtype=x.dtype,device=x.device).view(1,9,1)
    span=(maximum-minimum).clamp_min(1e-12)
    available=torch.stack([(x[:,i*3:i*3+3]!=0).flatten(1).any(1) for i in range(3)],1)
    native=(x-minimum)/span
    for i in range(3):native[:,i*3:i*3+3]=torch.where(
        available[:,i,None,None],native[:,i*3:i*3+3],torch.zeros_like(native[:,i*3:i*3+3]))
    assert torch.isfinite(native).all()
    return native,available

def _transpose_geometry(input_hw,target_hw,kernel):
    paddings,outputs=[],[]
    for size,target in zip(input_hw,target_hw):
        choices=[(p,o) for p in [1,0,2] for o in [0,1]
            if (size-1)*2-2*p+kernel+o==target]
        assert choices,(input_hw,target_hw,kernel)
        p,o=choices[0];paddings.append(p);outputs.append(o)
    return tuple(paddings),tuple(outputs)

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        widths=[1,64,128,256,512]
        self.encoder=nn.ModuleList([nn.Conv2d(widths[i],widths[i+1],5,stride=2,padding=2) for i in range(4)])
        shapes=[SHAPE]
        for _ in range(4):shapes.append(tuple((size+1)//2 for size in shapes[-1]))
        self.encoded_shape=(512,)+shapes[-1]
        flattened=int(np.prod(self.encoded_shape))
        self.enc_dense=nn.Linear(flattened,LATENT)
        self.dec_dense=nn.Linear(LATENT,flattened)
        self.decoder=nn.ModuleList()
        current=shapes[-1]
        for i,(kernel,target) in enumerate(zip([3,2,3,2],reversed(shapes[:-1]))):
            padding,output_padding=_transpose_geometry(current,target,kernel)
            self.decoder.append(nn.ConvTranspose2d(widths[4-i],widths[3-i],kernel,
                stride=2,padding=padding,output_padding=output_padding))
            current=target
        self.shape_metadata=dict(input_shape=SHAPE,encoded_shape=self.encoded_shape,latent=LATENT,
            encoder_kernels=[5]*4,decoder_kernels=[3,2,3,2],
            decoder_padding=[list(x.padding) for x in self.decoder],
            decoder_output_padding=[list(x.output_padding) for x in self.decoder])

    def encode(self,x):
        validate(x);values=x[:,None]
        for layer in self.encoder:values=F.relu(layer(values))
        assert tuple(values.shape[1:])==self.encoded_shape
        return self.enc_dense(values.flatten(1))

    def decode(self,z):
        values=F.relu(self.dec_dense(z)).view(len(z),*self.encoded_shape)
        for layer in self.decoder[:-1]:values=F.relu(layer(values))
        values=torch.sigmoid(self.decoder[-1](values))[:,0]
        assert tuple(values.shape[1:])==SHAPE and torch.isfinite(values).all()
        return values

    def forward(self,x):return self.decode(self.encode(x))

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv2d(1 if i==0 else 64,64,(1,5)) for i in range(4)])
        self.attention=nn.MultiheadAttention(embed_dim=9*64,num_heads=1)
        self.dropout=nn.Dropout(.5)
        self.fc=nn.Linear(9*64*(1024-4*(5-1)),6)

    def forward(self,x):
        validate(x);values=x[:,None]
        for layer in self.convs:values=F.relu(layer(values))
        # Official HAR layout: temporal sequence, batch, flattened filter×sensor.
        values=values.permute(3,0,1,2).reshape(1008,len(x),576)
        values,_=self.attention(values,values,values,need_weights=False)
        values=F.relu(self.dropout(values)).permute(1,0,2).reshape(len(x),-1)
        return self.fc(values)

def corrupt_mode4(native,rng):
    """Fixed Gaussian .1, independent-channel alternating exponential gaps."""
    assert native.device.type=='cpu'
    clean=native.detach().numpy()
    corrupted=clean+.1*rng.standard_normal(clean.shape).astype(np.float32)
    mask=np.ones(clean.shape,dtype=np.bool_)
    for sample in range(len(clean)):
        for channel in range(9):
            pointer=0;normal=True
            while pointer<1024:
                duration=max(1,int(np.rint(rng.exponential(480 if normal else 300))))
                end=min(1024,pointer+duration)
                if not normal:mask[sample,channel,pointer:end]=False
                pointer=end;normal=not normal
    corrupted[~mask]=0
    assert np.isfinite(corrupted).all()
    return torch.from_numpy(corrupted),torch.from_numpy(mask)

def parameter_counts(dae,classifier):
    return dict(dae=sum(p.numel() for p in dae.parameters()),
        classifier=sum(p.numel() for p in classifier.parameters()),
        total=sum(p.numel() for p in dae.parameters())+sum(p.numel() for p in classifier.parameters()))

def stage_seed(dataset, seed, stage):
    offsets = {'rfly': 0, 'uavfd': 200000}
    return int(seed) + offsets[dataset] + {'dae': 1100000, 'classifier': 2100000}[stage]


class Centaur(nn.Module):
    """Two separately trained stages; forward expects native min-max inputs."""
    def __init__(self, dae, classifier):
        super().__init__()
        self.dae = dae
        self.classifier = classifier

    def forward(self, native):
        return self.classifier(self.dae(native))


def build_centaur(num_classes, seed=42, dataset=None):
    # Reproduce the actual per-dataset stage initialization, including the
    # temporary six-class head used before the UAV-FD three-class replacement.
    if dataset is None:
        dataset = 'uavfd' if int(num_classes) == 3 else 'rfly'
    with cpu_seed(stage_seed(dataset, seed, 'dae')):
        dae = DAE()
    with cpu_seed(stage_seed(dataset, seed, 'classifier')):
        classifier = Classifier()
        if int(num_classes) != 6:
            classifier.fc = nn.Linear(classifier.fc.in_features, int(num_classes))
    return dae, classifier
