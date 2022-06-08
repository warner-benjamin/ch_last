import wandb
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

path = untar_data(URLs.PETS)/'images'

from fastcore.transform import DisplayedTransform, _is_tuple

class LoseTypeTransform(DisplayedTransform):
    "Does not retain_type"
    def __init__(self, **kwargs): super().__init__(**kwargs)

    def _call(self, fn, x, split_idx=None, **kwargs):
        if split_idx!=self.split_idx and self.split_idx is not None: return x
        return self._do_call(getattr(self, fn), x, **kwargs)

    def _do_call(self, f, x, **kwargs):
        if not _is_tuple(x):
            if f is None: return x
            return f(x, **kwargs)
        return tuple(self._do_call(f, x_, **kwargs) for x_ in x)
    
class ChannelsLastTfm(LoseTypeTransform):
    "Sets image-like inputs to `channels_last` format. For use in ChannelsLastCallback"
    order = 110 # run after all other transforms if added to batch_tfms
    def encodes(self, x:TensorImageBase|TensorMask):
        return x.to(memory_format=torch.channels_last)
    
class RemoveTensorType(LoseTypeTransform):
    order = 110 # run after all other transforms if added to batch_tfms
    def encodes(self, x:TensorImageBase|TensorMask):
        return torch.tensor(x)
    
class ChannelsLastCallback(Callback):
    "Channels last training using PyTorch's Channels Last Memory Format (beta)"
    order = MixedPrecision.order+1
    def __init__(self):
        self._channels_last = Pipeline([ChannelsLastTfm()])

    def before_fit(self):
        self.learn.model.to(memory_format=torch.channels_last)

    def before_batch(self):
        self.learn.xb = self._channels_last(self.xb)

def get_dls(bs, batch_tfms=None):
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=lambda x: x[0].isupper(), item_tfms=Resize(224), bs=bs, batch_tfms=batch_tfms)
    return dls

def train(loss_func, opt_func=Adam, channels_last=False, batch_size=128, epochs=2, tt=False):
    config = SimpleNamespace(channels_last=channels_last, 
                             batch_size=batch_size,
                             model_name="resnet50",
                             loss_func=loss_func,
                             opt_func=opt_func,
                             tt=tt,
                             )
    with wandb.init(project='channels_last', group="fastai", config=config):
        dls = get_dls(batch_size, batch_tfms=RemoveTensorType() if (tt or channels_last) else None)
        
        cbs = [WandbCallback(log_preds=False)]
        cbs += [ChannelsLastCallback()] if channels_last else []
        t0 = time.perf_counter()
        learn = vision_learner(dls, resnet50, 
                               loss_func=loss_func, opt_func=opt_func, 
                               metrics=error_rate, cbs=cbs, 
                               pretrained=False).to_fp16()
        wandb.summary["total_time"] = time.perf_counter() - t0
        learn.fit(epochs)
    return

N = 2
bs=128

for _ in range(N):
    for tt in [True, False]:
        for loss_func in [nn.CrossEntropyLoss(), CrossEntropyLossFlat()]:
            for channels_last in [True, False]:
                train(loss_func, channels_last=channels_last,  batch_size=bs, tt=tt)