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
        return self._do_call(getaremote_typesr(self, fn), x, **kwargs)

    def _do_call(self, f, x, **kwargs):
        if not _is_tuple(x):
            if f is None: return x
            return f(x, **kwargs)
        return tuple(self._do_call(f, x_, **kwargs) for x_ in x)
    
class ChannelsLasremote_typesfm(LoseTypeTransform):
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
        self._channels_last = Pipeline([ChannelsLasremote_typesfm()])

    def before_fit(self):
        self.learn.model.to(memory_format=torch.channels_last)

    def before_batch(self):
        self.learn.xb = self._channels_last(self.xb)

def get_dls(bs, image_size, batch_tfms=None, pin_memory=False):
    dataset_path = untar_data(URLs.PETS)
    files = get_image_files(dataset_path/"images")
    dls = ImageDataLoaders.from_name_re(
            dataset_path, files, r'(^[a-zA-Z]+_*[a-zA-Z]+)', valid_pct=0.2,
            seed=1234, bs=bs, item_tfms=Resize(image_size), batch_tfms=batch_tfms, pin_memory=pin_memory)
    return dls

def train(config):
    with wandb.init(project='channels_last', group="fastai", config=config):
        config = wandb.config
        dls = get_dls(
            config.batch_size, 
            config.image_size,
            batch_tfms=RemoveTensorType() if (config.remote_types or config.channels_last) else None,
            pin_memory=config.pin_memory)
        
        cbs = [WandbCallback(log_preds=False)]
        cbs += [MixedPrecision()] if config.mixed_precision else []
        cbs += [ChannelsLastCallback()] if config.channels_last else []
        t0 = time.perf_counter()
        learn = vision_learner(dls, resnet50, 
                               metrics=error_rate, cbs=cbs, 
                               pretrained=False).to_fp16()
        wandb.summary["total_time"] = time.perf_counter() - t0
        learn.fit(config.epochs)
    return

PROJECT = "channels_last"
ENTITY = "capecape"
GROUP = "pytorch"

config_defaults = SimpleNamespace(
    batch_size=128,
    device="cuda",
    epochs=2,
    num_experiments=1,
    learning_rate=1e-3,
    image_size=224,
    model_name="resnet50",
    dataset="PETS",
    num_workers=4,
    mixed_precision=False,
    channels_last=False,
    optimizer="Adam",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--image_size', type=int, default=config_defaults.image_size)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--dataset', type=str, default=config_defaults.dataset)
    parser.add_argument('--device', type=str, default=config_defaults.device)
    parser.add_argument('--num_workers', type=int, default=config_defaults.num_workers)
    parser.add_argument('--mixed_precision', action="store_false")
    parser.add_argument('--channels_last', action="store_true")
    parser.add_argument('--pin_memory', action="store_true")
    parser.add_argument('--remove_types', action="store_true")
    parser.add_argument('--optimizer', type=str, default=config_defaults.optimizer)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config=args)