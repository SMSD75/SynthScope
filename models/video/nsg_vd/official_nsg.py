"""NSG-VD's feature extractor: the Normalized Spatiotemporal Gradient (NSG) of a clip, ported from the official code
(github.com/ZSHsh98/NSG-VD @ 9a60f57; Apache-2.0, Copyright the NSG-VD authors) with the same operations in the same
order. Sources:

* ``UNET_CONFIG`` / ``build_unet``: the OpenAI guided-diffusion 256x256 unconditional model as the official
  ``RevGuidedDiffusion`` builds it (libs/eps_ad/runners/diffpure_sde.py: ``model_and_diffusion_defaults()`` updated with
  the ``model`` block of libs/eps_ad/imagenet.yml, then ``create_model``; libs/eps_ad/guided_diffusion/script_util.py),
  weights ``256x256_diffusion_uncond.pt`` loaded strictly, then ``convert_to_fp16()`` (``use_fp16: True``).
* ``score_fn``: libs/eps_ad/score_sde/models/utils.py ``get_score_fn(VPSDE(0.1, 20, 1000), model, train=False,
  continuous=True)`` (data/utils.py ``get_score_fn``): ``-model(x, 999 t) / std(t)``.
* ``score_velocity`` and ``score_features``: data/feature_dataset/score_feature_dataset.py ``_get_score_velocity`` and
  ``_get_score_features`` (methods there, functions here; ``self.*`` became arguments).
* ``scale_end_frames``: data/feature_dataset/score_feature_dataset.py ``get_data`` (feature_type "velocity",
  filter_frames False): the first and last frames of the cached NSG are multiplied by 100 when loaded.
"""
import torch
import torch.nn.functional as F

from models.video.nsg_vd.guided_diffusion.unet import UNetModel

# model_and_diffusion_defaults() (guided_diffusion/script_util.py) updated with imagenet.yml's model block, restricted to
# the create_model arguments; create_model's derived values written out (image_size 256 -> channel_mult (1, 1, 2, 2,
# 4, 4); attention_resolutions "32,16,8" -> downsample rates (8, 16, 32); learn_sigma -> 6 output channels).
UNET_CONFIG = dict(
    image_size=256, in_channels=3, model_channels=256, out_channels=6, num_res_blocks=2,
    attention_resolutions=(256 // 32, 256 // 16, 256 // 8), dropout=0.0, channel_mult=(1, 1, 2, 2, 4, 4),
    num_classes=None, use_checkpoint=False, use_fp16=True, num_heads=4, num_head_channels=64, num_heads_upsample=-1,
    use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=False,
)
BETA_MIN, BETA_MAX = 0.1, 20       # RevVPSDE defaults, passed to sde_lib.VPSDE by data/utils.py get_score_fn


def build_unet(**overrides) -> UNetModel:
    """The official diffusion UNet (random weights; load ``256x256_diffusion_uncond.pt`` into it)."""
    return UNetModel(**{**UNET_CONFIG, **overrides})


def vpsde_std(t: torch.Tensor) -> torch.Tensor:
    """sde_lib.VPSDE.marginal_prob(x, t)[1] (beta_0 = 0.1, beta_1 = 20)."""
    log_mean_coeff = -0.25 * t ** 2 * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    return torch.sqrt(1. - torch.exp(2. * log_mean_coeff))


def score_fn(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """score_sde get_score_fn for a VPSDE, continuous=True, train=False."""
    labels = t * 999
    model.eval()
    score = model(x, labels)
    std = vpsde_std(t)
    score = -score / std[:, None, None, None]
    return score


@torch.no_grad()
def score_velocity(score, image, epsilon=1e-10):
    """
    args:
        score: #(B,T,C,H,W)
        image: #(B,T,C,H,W)
    return:
        velocity: #(B,T,C,H,W)
    """
    B, T, C, H, W = score.shape

    pixel_diff = torch.zeros((B, T, C, H, W))  # (B,T,C,H,W)
    for t in range(1, T - 1):
        pixel_diff[:, t] = (image[:, t + 1, ...] - image[:, t - 1, ...]) / 2  # (B,C,H,W)
    pixel_diff[:, 0] = (image[:, 1, ...] - image[:, 0, ...])  # (B,C,H,W)
    pixel_diff[:, -1] = (image[:, -1, ...] - image[:, -2, ...])  # (B,C,H,W)
    pixel_diff = pixel_diff.to(score.device)

    numerator = score  # (B,T,C,H,W)
    denominator = (score * pixel_diff).sum(dim=(2, 3, 4)).view(B, T, 1, 1, 1) + epsilon  # (B,T,1,1,1)
    assert denominator.shape == (B, T, 1, 1, 1)
    velocity = numerator / denominator
    assert velocity.shape == (B, T, C, H, W)
    return velocity


@torch.no_grad()
def score_features(model, frame_tensor, batch_size, diffuse_steps=5, resolution_size=224, input_shape=(224, 224)):
    """
    args:
        frame_tensor: (B*T,C,H,W) frames in [0, 1]
        batch_size: B
    return:
        scores: (B,T,C,H,W)
        velocity: (B,T,C,H,W)
    """
    t_value = (diffuse_steps + 1) / 1000
    curr_t_temp = torch.tensor(t_value, device=frame_tensor.device)

    image = frame_tensor.view(batch_size, -1, 3, 224, 224)
    frame_tensor = F.interpolate(frame_tensor, size=(resolution_size, resolution_size), mode='bilinear',
                                 align_corners=False)  # (B,T,C,H,W)
    score = score_fn(model, 2 * frame_tensor - 1, curr_t_temp.expand(frame_tensor.shape[0]))
    score, _ = torch.split(score, score.shape[1] // 2, dim=1)  # (B*T,C,H,W)

    score = F.interpolate(score, size=(input_shape[0], input_shape[1]), mode='bilinear', align_corners=False)

    image = image.view(batch_size, -1, *score.shape[1:])  # (B,T,C,H,W)
    score = score.view(batch_size, -1, *score.shape[1:])  # (B,T,C,H,W)

    velocity = score_velocity(score, image)  # (B,T,C,H,W)

    return score, velocity


def scale_end_frames(velocity: torch.Tensor) -> torch.Tensor:
    """get_data (feature_type "velocity"): ``velocity[0] *= 100; velocity[-1] *= 100`` on each clip's (T,C,H,W) NSG,
    here on a [B, T, C, H, W] batch (a copy)."""
    velocity = velocity.clone()
    velocity[:, 0] = velocity[:, 0] * 100
    velocity[:, -1] = velocity[:, -1] * 100
    return velocity
