# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from misc.detr_utils import box_ops
from misc.detr_utils.misc import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """Compute DETR-style matching losses for SAVA-X."""

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2, opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.opt = opt
        counter_class_rate = [
            0.0,
            0.0,
            0.004186728072011723,
            0.31034121833786893,
            0.30584048566045635,
            0.1692484823110739,
            0.11586769939292443,
            0.04940339124973833,
            0.024387691019468284,
            0.010466820180029307,
            0.005966087502616705,
            0.0020933640360058614,
            0.0017793594306049821,
            0.00020933640360058616,
            0.0,
            0.00020933640360058616,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.counter_class_rate = torch.tensor(counter_class_rate)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification losses for event type, fine error, overall error, and count."""
        indices, many2one_indices = indices
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=0.75,
            gamma=1,
        ) * src_logits.shape[1]
        losses = {"loss_ce": loss_ce}

        pred_mimic_fine = outputs["mimic_fine_logits"][idx]
        target_fine_o = torch.cat([t["mimic_fine_label"][J] for t, (_, J) in zip(targets, indices)])
        target_fine = target_fine_o.unsqueeze(-1).type_as(pred_mimic_fine)
        losses["loss_mimic_fine"] = sigmoid_focal_loss(
            pred_mimic_fine,
            target_fine,
            1,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        pred_mimic_overall = outputs["mimic_overall_logits"]
        target_overall = torch.tensor(
            [t["mimic_overall_label"] for t in targets],
            device=pred_mimic_overall.device,
            dtype=pred_mimic_overall.dtype,
        ).view(-1, 1)
        losses["loss_mimic_overall"] = sigmoid_focal_loss(
            pred_mimic_overall,
            target_overall,
            1,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        pred_count = outputs["pred_count"]
        max_length = pred_count.shape[1] - 1
        counter_target = [
            len(target["boxes"]) if len(target["boxes"]) < max_length else max_length
            for target in targets
        ]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count)
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1)
        weight = self.counter_class_rate[: max_length + 1].to(src_logits.device)
        losses["loss_counter"] = cross_entropy_with_gaussian_mask(
            pred_count,
            counter_target_onehot,
            self.opt,
            weight,
        )

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Cardinality error for logging only."""
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Box regression and generalized IoU losses."""
        indices, many2one_indices = indices
        assert "pred_boxes" in outputs
        idx, idx2 = self._get_src_permutation_idx2(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cl_to_xy(src_boxes),
                box_ops.box_cl_to_xy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        self_iou = torch.triu(
            box_ops.box_iou(
                box_ops.box_cl_to_xy(src_boxes),
                box_ops.box_cl_to_xy(src_boxes),
            )[0],
            diagonal=1,
        )
        sizes = [len(v[0]) for v in indices]
        self_iou_split = 0
        for i, c in enumerate(self_iou.split(sizes, -1)):
            cc = c.split(sizes, -2)[i]
            self_iou_split += cc.sum() / (0.5 * sizes[i] * (sizes[i] - 1))
        losses["loss_self_iou"] = self_iou_split

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx2(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_idx2 = torch.cat([src for (_, src) in indices])
        return (batch_idx, src_idx), src_idx2

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """Perform the loss computation."""
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        last_indices = self.matcher(outputs_without_aux, targets)
        outputs["matched_indices"] = last_indices

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes))

        if "aux_outputs" in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                aux_indices.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs["log"] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            return losses, last_indices, aux_indices
        return losses, last_indices


def cross_entropy_with_gaussian_mask(inputs, targets, opt, weight):
    """Counter loss with the original Gaussian neighborhood weighting."""
    gau_mask = getattr(opt, "lloss_gau_mask", 1)
    beta = getattr(opt, "lloss_beta", 1)

    _, max_seq_len = targets.shape
    gaussian_mu = torch.arange(max_seq_len, device=inputs.device).unsqueeze(0).expand(
        max_seq_len, max_seq_len
    ).float()
    x = gaussian_mu.transpose(0, 1)
    gaussian_sigma = 2
    mask_dict = torch.exp(-((x - gaussian_mu) ** 2) / (2 * gaussian_sigma ** 2))
    _, ind = targets.max(dim=1)
    mask = mask_dict[ind]

    loss = F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
        weight=1 - weight,
    )
    if gau_mask:
        coef = targets + ((1 - mask) ** beta) * (1 - targets)
    else:
        coef = targets + (1 - targets)
    loss = loss * coef
    loss = loss.mean(1)
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """Focal BCE used by the detection and mimic heads."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean() / num_boxes
