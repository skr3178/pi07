"""Matched SmolVLA policy variants with explicit, maskable visual-goal tokens.

Both A and B use the same observation/goal/text attention convention. Only B
receives valid goal tokens. No new learned weights are introduced by the wrapper.
"""
import math

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

CAMERAS = ("observation.images.camera1", "observation.images.camera2")
GOALS = ("observation.images.goal_camera1", "observation.images.goal_camera2")


def configure_goals(config):
    """Use an identical four-slot interface for the matched A and B policies."""
    if config.add_image_special_tokens or config.empty_cameras or config.prefix_length > 0:
        raise ValueError("This tested adapter uses no image-special tokens, empty cameras or fixed prefix length")
    for key in CAMERAS:
        if key not in config.input_features:
            raise ValueError(f"Required current camera missing: {key}")
    for current, goal in zip(CAMERAS, GOALS):
        config.input_features[goal] = PolicyFeature(type=FeatureType.VISUAL,
                                                   shape=config.input_features[current].shape)
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.train_state_proj = True
    return config


class GoalFlowMatching(VLAFlowMatching):
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        if len(images) != 4 or len(img_masks) != 4:
            raise ValueError("Expected two current and two goal camera slots")
        embs, pads, blocks = [], [], []
        for img, mask in zip(images, img_masks):
            # Keep a fixed layout in A, B and dropped-goal evaluation. Removing
            # slots changed numerical outputs in the integration check. Masks
            # exclude hidden goals from every valid query's attention.
            value = self.vlm_with_expert.embed_image(img)
            value = value * math.sqrt(value.shape[-1])
            embs.append(value)
            pads.append(mask[:, None].expand(value.shape[:2]))
            blocks.extend([0] * value.shape[1])
        obs_tokens = sum(x.shape[1] for x in embs[:2])
        lang = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        embs.append(lang * math.sqrt(lang.shape[-1])); pads.append(lang_masks)
        blocks.extend([0] * lang.shape[1])
        # Current observations see only observations. Goals and text share the
        # next block and can see observations. This boundary also exists in A.
        blocks[obs_tokens] = 1
        st = self.state_proj(state)[:, None, :]
        embs.append(st)
        pads.append(torch.ones(st.shape[:2], dtype=torch.bool, device=st.device))
        blocks.append(1)
        return (torch.cat(embs, 1), torch.cat(pads, 1),
                torch.tensor(blocks, dtype=torch.bool, device=st.device)[None].expand(st.shape[0], -1))


class GoalSmolVLAPolicy(SmolVLAPolicy):
    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)
        self.config = configure_goals(config)
        self.model = GoalFlowMatching(self.config)
        self.reset()

    def prepare_images(self, batch):
        if not all(key in batch for key in (*CAMERAS, *GOALS)):
            raise ValueError("Supply both current images and both goal slots, even when goals are dropped")
        for key in GOALS:
            if f"{key}_padding_mask" not in batch:
                raise ValueError("Every goal must have an explicit validity mask")
        if not torch.equal(batch[f"{GOALS[0]}_padding_mask"], batch[f"{GOALS[1]}_padding_mask"]):
            raise ValueError("Main and wrist goals must share a validity mask")
        # Parent uses config key order, which is validated here rather than inferred.
        if tuple(self.config.image_features) != (*CAMERAS, *GOALS):
            raise ValueError("Unexpected image-token ordering")
        return super().prepare_images(batch)

    def forward(self, batch, noise=None, time=None):
        images, masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        targets = self.prepare_action(batch)
        is_pad = batch.get("action_is_pad")
        if is_pad is None:
            raise ValueError("Training requires action_is_pad to exclude episode-end padding")
        is_pad = is_pad.to(device=targets.device, dtype=torch.bool)
        # Canonicalize padded targets before noising, so changing their values
        # cannot indirectly influence valid actions through model attention.
        targets = targets.masked_fill(is_pad[..., None], 0)
        loss_per_component = self.model.forward(
            images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            state, targets, noise, time,
        )
        action_dim = self.config.action_feature.shape[0]
        valid = (~is_pad)[..., None].expand(-1, -1, action_dim)
        count = valid.sum()
        if not count:
            raise ValueError("Batch has no valid action targets")
        loss = loss_per_component[..., :action_dim].masked_select(valid).mean()
        return loss, {"loss": float(loss.detach()), "valid_action_components": int(count)}
