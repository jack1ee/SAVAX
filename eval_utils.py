from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import torch
import numpy as np
import json
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
from os.path import dirname, abspath

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))


from densevid_eval3.eval_soda import eval_soda
from densevid_eval3.eval_para import eval_para
from densevid_eval3.eval_dvc import eval_dvc
from densevid_eval3.eval_imitation import eval_imitation

import re
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    _USE_SNS = True
except Exception:
    _USE_SNS = False

def calculate_avg_proposal_num(json_path):
    data = json.load(open(json_path))
    return np.array([len(v) for v in data['results'].values()]).mean()

def convert_tapjson_to_dvcjson(tap_json, dvc_json):
    data = json.load(open(tap_json, 'r'))
    data['version'] = "VERSION 1.0"
    data['external_data'] = {'used:': True, 'details': "C3D pretrained on Sports-1M"}

    all_names = list(data['results'].keys())
    for video_name in all_names:
        for p_info in data['results'][video_name]:
            p_info['timestamp'] = p_info.pop('segment')
            p_info['proposal_score'] = p_info.pop('score')
            p_info['sentence_score'] = p_info.pop('sentence_score', 0)
        data['results']["v_" + video_name] = data['results'].pop(video_name)
    with open(dvc_json, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())


def convert_dvcjson_to_tapjson(dvc_json, tap_json):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        event_num = len(data[video_name])
        timestamps = [data[video_name][i]['timestamp'] for i in range(event_num)]
        sentences = [data[video_name][i]['sentence'] for i in range(event_num)]
        for i, timestamp in enumerate(timestamps):
            score = data[video_name][i].get('proposal_score', 1.0)
            video_info.append({'segment': timestamp, 'score': score, 'sentence': sentences[i], 'sentence_score': data[video_name][i].get('sentence_score', 0)})
        out['results'][video_name[2:]] = video_info
    with open(tap_json, "w") as f:
        json.dump(out, f)
        f.flush()
        os.fsync(f.fileno())


def convert_gtjson_to_tapjson(gt_json, tap_json):
    data = json.load(open(gt_json, 'r'))
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        timestamps = data[video_name]['timestamps']
        sentences = data[video_name]['sentences']
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1., 'sentence': sentences[i]})
        out['results'][video_name[2:]] = video_info
    with open(tap_json, 'w') as f:
        json.dump(out, f)
        f.flush()
        os.fsync(f.fileno())

def get_topn_from_dvcjson(dvc_json, out_json, top_n=3, ranking_key='proposal_score', score_thres=-1e8):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}
    all_names = list(data.keys())
    num = 0
    bad_vid = 0
    for video_name in all_names:
        info = data[video_name]
        new_info = sorted(info, key=lambda x: x[ranking_key], reverse=True)
        new_info = [p for p in new_info if p[ranking_key] > score_thres]
        new_info = new_info[:top_n]
        out['results'][video_name] = new_info
        num += len(new_info)
        if len(new_info) == 0:
            bad_vid += 1
            out['results'].pop(video_name)
    print('average proosal number: {}'.format(num / len(all_names)))
    print('bad videos number: {}'.format(bad_vid))
    print('good videos number: {}'.format(len(out['results'])))
    with open(out_json, 'w') as f:
        json.dump(out, f)
        f.flush()
        os.fsync(f.fileno())


def eval_metrics(dvc_filename, gt_filenames, para_gt_filenames, alpha=0.3, ranking_key='proposal_score', rerank=False, dvc_eval_version='2018',w_correct=0.2,
                 phase='val',
                 num_thresh=100,
                 thresh_sampling='quantile',
                 fix_fine_thresh=True,
                 global_fine_thresh=None,
                 global_overall_thresh=None,
                 tiou_for_thresh=None):
    score = collections.defaultdict(lambda: -1)

    # top_n = 3
    # top_n_filename = dvc_filename + '.top{}.json'.format(top_n)
    # get_topn_from_dvcjson(dvc_filename, top_n_filename, top_n=top_n, ranking_key=ranking_key)
    # dvc_score = eval_dvc(json_path=top_n_filename, reference=gt_filenames)
    # dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    # dvc_score.update(eval_soda(top_n_filename, ref_list=gt_filenames))
    # dvc_score.update(eval_para(top_n_filename, referneces=para_gt_filenames))
    # for key in dvc_score.keys():
    #     score[key] = dvc_score[key]

    if rerank:
        dvc_filename = reranking(dvc_filename, alpha=alpha, temperature=2.0)
    dvc_score = eval_dvc(json_path=dvc_filename, reference=gt_filenames)
    dvc_score_mean = {
        k + '_mean': (sum(v) / len(v))
        for k, v in dvc_score.items()
        if isinstance(v, list) and len(v) != 1 and k not in ['fine_pr_curve_correct', 'fine_pr_curve_error']
        }
    dvc_score.update(dvc_score_mean)
    
    imitation_score = eval_imitation(json_path=dvc_filename, reference=gt_filenames, w_correct=w_correct,
                         phase=phase,
                         num_thresh=num_thresh,
                         thresh_sampling=thresh_sampling,
                         fix_fine_thresh=fix_fine_thresh,
                         global_fine_thresh=global_fine_thresh,
                         tiou_for_thresh=tiou_for_thresh,
                         global_overall_thresh=global_overall_thresh)
    dvc_score.update(imitation_score)
    imitation_score_mean = {
        k + '_mean': (sum(v) / len(v))
        for k, v in imitation_score.items()
        if isinstance(v, list) and len(v) != 1 and k not in ['fine_pr_curve_correct', 'fine_pr_curve_error']
        }
    imitation_score_mean.pop('iou_thres_mean')
    dvc_score.update(imitation_score_mean)
    
    dvc_score.update(eval_soda(dvc_filename, ref_list=gt_filenames))
    # Skip paragraph evaluation in this path.
    if dvc_score['CIDEr_mean']!=0:
        dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
    score.update(dvc_score)
    return score

def save_dvc_json(out_json, path):
    out_json['valid_video_num'] = len(out_json['results'])
    out_json['avg_proposal_num'] = np.array([len(v) for v in out_json['results'].values()]).mean().item()
    with open(path, 'w') as f:
        json.dump(out_json, f)

def reranking(p_src, alpha, temperature):
    print('alpha: {}, temp: {}'.format(alpha, temperature))
    d = json.load(open(p_src))
    d_items = list(d['results'].items())
    for k,v in d_items:
        if True:
            sent_scores = [p['sentence_score'] / (float(len(p['sentence'].split()))**(temperature) + 1e-5) for p in v]
            prop_score = [p['proposal_score'] for p in v]
            joint_score = alpha * (np.array(sent_scores)) + (np.array(prop_score))
        for i,p in enumerate(v):
            p['joint_score'] = joint_score[i]
        v = sorted(v, key=lambda x: x['joint_score'], reverse=True)
        topN = v[0]['pred_event_count']
        v = v[:topN]
        v = sorted(v, key=lambda x: x['timestamp'])
        d['results'][k] = v
    root, ext = os.path.splitext(p_src)  # Strip only the last extension.
    save_path = root+'_rerank_alpha{}_temp{}.json'.format(alpha, temperature)
    save_dvc_json(d, save_path)
    return save_path


def evaluate(model, criterion, postprocessors, loader, dvc_json_path, logger=None, score_threshold=0,
             alpha=0.3, dvc_eval_version='2018', device='cuda', debug=False, skip_lang_eval=False, save_feat=False, w_correct=0.2,
             eval_phase=None,                  # 'val' or 'test'; infer from the dataset split if omitted.
             tiou_for_thresh=0.5,              # Validation tIoU used to pick a global fine threshold.
             num_thresh=100,
             thresh_sampling='quantile',
             fix_fine_thresh=True,
             fixed_fine_thresh=None,
             fixed_overall_thresh=None,
             vis_out_dir=None,             # Visualization output directory.
             vis_num=0,                    # Max number of saved visualizations; 0 disables it.
             vis_logit_norm='none',
             vis_keys=None):
    out_json = {'results': {},
                'version': "VERSION 1.0",
                'external_data': {'used:': True, 'details': None}}
    opt = loader.dataset.opt
    loss_sum = OrderedDict()
    n_frame_feat = 0
    sum_view_acc = 0
    sum_mimic_overall_acc=0
    if save_feat:
        input_feat_list = []
        converted_feat_list = []
        view_labels_list = []
    with torch.set_grad_enabled(False):
        _viz_saved = 0
        target_keys = set(vis_keys) if vis_keys else None
        for dt in tqdm(loader, disable=opt.disable_tqdm):
            # valid_keys = ["video_tensor", "video_length", "video_mask", "video_key"]
            # dt = {key: value for key, value in dt.items() if key in valid_keys}
            dt = {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt = collections.defaultdict(lambda: None, dt)

            dt['video_target'] = [
                    {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                    dt['video_target']]

            output, loss = model(dt, criterion, opt.transformer_input_type, eval_mode=True)
            
            mimic_fine_logits=output['mimic_fine_logits']
            mimic_overall_logits=output['mimic_overall_logits']

            if save_feat:
                input_feat = dt['video_tensor'].cpu().numpy() # 1, 200, X
                assert input_feat.shape[0] == 1 and input_feat.shape[1] == 200, f"input_feat shape is {input_feat.shape}"
                converted_feat = output["out_feat"].cpu().numpy() # 1, 200, 1024
                view_labels = output["view_labels"].cpu().numpy() # 1, 200, 1
                assert view_labels.shape[0] == 1 and view_labels.shape[1] == 200 and view_labels.shape[2] == 1, f"view_labels shape is {view_labels.shape}"
                input_feat_list.append(input_feat)
                converted_feat_list.append(converted_feat)
                view_labels_list.append(view_labels)
                
            orig_target_sizes = dt['video_length'][:, 1]
            # print(f"==>> dt['video_length'].shape: {dt['video_length'].shape}")
            
            # n_frame_feat += int(dt['video_length'][:, 0])
            
            # print("n_frame_feat: ", n_frame_feat)
            # print("output: ", output['view_acc'])
            if 'view_acc' in output:
                if output['view_acc'] >= 0:
                    sum_view_acc += float(output['view_acc'])


            weight_dict = criterion.weight_dict
            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)

            for loss_k, loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0) + loss_v.item()
            try:
                loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()
            except:
                loss_sum['total_loss'] = 0 
            results = postprocessors['bbox'](output, orig_target_sizes, loader)

            batch_json = {}
            for idx, video_name in enumerate(dt['video_key']):
                segment = results[idx]['boxes'].cpu().numpy()
                raw_boxes = results[idx]['raw_boxes'].cpu().numpy()
                topk_mimic_fine_scores = results[idx].get('mimic_fine_scores', None)
                # pdb.set_trace()
                batch_json[video_name] = [
                    {
                        "timestamp": segment[pid].tolist(),
                        "raw_box": raw_boxes[pid].tolist(),
                        "proposal_score": results[idx]['scores'][pid].item(),
                        # Prefer top-k aligned scores from PostProcess; fallback by query_id.
                        "mimic_fine_score": (
                            topk_mimic_fine_scores[pid].item()
                            if topk_mimic_fine_scores is not None
                            else mimic_fine_logits.sigmoid()[idx][results[idx]['query_id'][pid]].item()
                        ),
                        "mimic_overall_score": mimic_overall_logits.sigmoid()[idx].item(),
                        "sentence": results[idx]['captions'][pid],
                        "sentence_score": results[idx]['caption_scores'][pid],
                        'query_id': results[idx]['query_id'][pid].item(),
                        'vid_duration': results[idx]['vid_duration'].item(),
                        'pred_event_count': results[idx]['pred_seq_len'].item(),
                    }
                    for pid in range(len(segment)) if results[idx]['scores'][pid].item() > score_threshold]
            out_json['results'].update(batch_json)
            if debug and len(out_json['results']) > 5:
                break
            
            # Save a small sample of cached logits and attention maps.
            if vis_out_dir:
                try:
                    if target_keys is not None and len(target_keys) > 0:
                        _viz_saved += _save_logits_viz(dt, model, vis_out_dir, vis_num - _viz_saved, norm_mode=vis_logit_norm,
                                                       filter_keys=target_keys)
                    elif target_keys is None and vis_num > 0 and _viz_saved < vis_num:
                        _viz_saved += _save_logits_viz(dt, model, vis_out_dir, vis_num - _viz_saved, norm_mode=vis_logit_norm)
                except Exception as e:
                    if logger: logger.warning(f"[viz-skip] {e}")

    save_dvc_json(out_json, dvc_json_path)

    if skip_lang_eval:
        return None, None

    for k in loss_sum.keys():
        loss_sum[k] = np.round(loss_sum[k] / (len(loader) + 1e-5), 3).item()
    logger.info('loss: {}'.format(loss_sum))
    print("eval files", opt.gt_file_for_eval, opt.gt_file_for_para_eval)
    scores = eval_metrics(dvc_json_path,
                            gt_filenames=opt.gt_file_for_eval,
                            para_gt_filenames=opt.gt_file_for_para_eval,
                            alpha=alpha,
                            rerank=(opt.count_loss_coef > 0),
                            dvc_eval_version=dvc_eval_version,
                            w_correct = w_correct,
                            phase=eval_phase,
                            num_thresh=num_thresh,
                            thresh_sampling=thresh_sampling,
                            fix_fine_thresh=fix_fine_thresh,
                            global_fine_thresh=fixed_fine_thresh,
                            tiou_for_thresh=tiou_for_thresh,
                            global_overall_thresh=fixed_overall_thresh
                        )
    # scores['view_acc'] = sum_view_acc / n_frame_feat
    # scores['mimic_overall_acc'] = sum_mimic_overall_acc / len(loader)
    out_json.update(scores)
    save_dvc_json(out_json, dvc_json_path)
    if save_feat:        
        save_path = os.path.join(os.path.dirname(dvc_json_path), f"{loader.dataset.data_type}_feats")
        os.makedirs(save_path, exist_ok=True)
        print("feature save_path: ", save_path)
        input_feat_list = np.concatenate(input_feat_list, axis=0)
        converted_feat_list = np.concatenate(converted_feat_list, axis=0)
        view_labels_list = np.concatenate(view_labels_list, axis=0)
        print("input_feat_list: ", input_feat_list.shape)
        print("converted_feat_list: ", converted_feat_list.shape)
        print("view_labels_list: ", view_labels_list.shape)
        # save feat
        np.save(os.path.join(save_path, "input_feat.npy"), input_feat_list)
        np.save(os.path.join(save_path, "converted_feat.npy"), converted_feat_list)
        np.save(os.path.join(save_path, "view_labels.npy"), view_labels_list)
        
        
    return scores, loss_sum
# =================== Logit Visualization ===================
def _to_np(x): return x.detach().cpu().numpy()

# Map a sequence length to a uniform [0, 1] axis.
def _unit_x(L: int):
    if L <= 1:
        return np.array([0.0]) if L == 1 else np.array([])
    return np.linspace(0.0, 1.0, num=L)

def _save_logits_viz(dt, model, out_dir, remain, norm_mode='none',
                     filter_keys=None):
    if remain <= 0:
        return 0
    if not hasattr(model, 'sampler_ego') or model.sampler_ego is None:
        return 0
    os.makedirs(out_dir, exist_ok=True)

    keys = dt['video_key']                                        # list[str]
    ego_log = getattr(model.sampler_ego, 'last_logits_coarse', None)      # (B, Lq_ego)
    ego_msk = getattr(model.sampler_ego, 'last_mask_coarse', None)        # (B, Lq_ego)
    attn    = getattr(model.sampler_ego, 'last_cross_attn', None)         # (B, H, Lq_ego, Lk_sel)

    sam_exo = getattr(model, 'sampler_exo', None)
    exo_log = getattr(sam_exo, 'last_logits_coarse', None) if sam_exo is not None else None
    exo_msk = getattr(sam_exo, 'last_mask_coarse', None) if sam_exo is not None else None

    key_idx20 = None
    if sam_exo is not None and hasattr(sam_exo, 'last_index_map'):
        key_idx20 = sam_exo.last_index_map  # (B, Lk_pad) column-wise indices on the original 20 fps grid.

    if ego_log is None or ego_msk is None:
        return 0

    if _USE_SNS:
        sns.set_theme(style='darkgrid', context='paper')

    saved = 0
    B = ego_log.size(0)
    for b in range(B):
        raw_key = keys[b]
        if filter_keys is not None and raw_key not in filter_keys:
            continue
        safe_key = re.sub(r'[^a-zA-Z0-9_\\-]+', '_', raw_key)
        if saved >= remain:
            break

        name = re.sub(r'[^a-zA-Z0-9_\\-]+', '_', keys[b])
        Lq_ego = int(ego_msk[b].sum().item()) if ego_msk is not None else ego_log.size(1)
        y_ego_t = ego_log[b][:Lq_ego]
        y_ego = _to_np(_viz_normalize(y_ego_t, norm_mode))
        x_ego   = _unit_x(Lq_ego)  # Normalize the x-axis to [0, 1].

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), gridspec_kw={'height_ratios':[0.5,0.5,1.5]})

        # (1) Ego logits as a single-row heatmap.
        ax = axes[0]
        score_h = 6
        score_w = 100
        score_vmin = -0.3
        score_vmax = 1.5
        if Lq_ego > 0:
            y1d = torch.from_numpy(y_ego).float().view(1,1,-1)              # [1,1,Len]
            yw  = F.interpolate(y1d, size=score_w, mode='linear', align_corners=False)[0,0]  # [W]
            Y2d = yw.unsqueeze(0).repeat(score_h, 1).numpy()                # [H,W]
            im_ego = ax.imshow(Y2d, origin='lower', aspect='auto',
                               extent=[0.0, 1.0, 0.0, 1.0],
                               interpolation='nearest', vmin=score_vmin, vmax=score_vmax, cmap="YlOrRd")
            ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
            ax.set_title(f'Ego score heatmap (norm={norm_mode}, {score_h}×{score_w}, vmax={score_vmax})')
            ax.set_xlabel('normalized time (0–1)')
            ax.set_yticks([])  # A single-row heatmap reads better without y ticks.
            ax.set_xticks([])
        else:
            ax.text(0.5, 0.5, 'no ego scores', ha='center', va='center'); ax.axis('off')

        # (2) Exo logits, if cached.
        attn_ax_idx = 2
        if exo_log is not None and exo_msk is not None:
            Lq_exo = int(exo_msk[b].sum().item())
            y_exo_t = exo_log[b][:Lq_exo]
            y_exo = _to_np(_viz_normalize(y_exo_t, norm_mode))
            x_exo   = _unit_x(Lq_exo)  # Normalize the x-axis to [0, 1].
            ax2 = axes[1]
            if Lq_exo > 0:
                x1d = torch.from_numpy(y_exo).float().view(1,1,-1)          # [1,1,Len]
                xw  = F.interpolate(x1d, size=score_w, mode='linear', align_corners=False)[0,0]  # [W]
                X2d = xw.unsqueeze(0).repeat(score_h, 1).numpy()            # [H,W]
                ax2.imshow(X2d, origin='lower', aspect='auto',
                           extent=[0.0, 1.0, 0.0, 1.0],
                           interpolation='nearest', vmin=score_vmin, vmax=score_vmax, cmap="YlOrRd")
                ax2.set_xlim(0.0, 1.0); ax2.set_ylim(0.0, 1.0)
                ax2.set_title(f'Exo score heatmap (norm={norm_mode}, {score_h}×{score_w}, vmax={score_vmax})')
                ax2.set_xlabel('normalized time (0–1)')
                ax2.set_yticks([])
                ax2.set_xticks([])
            else:
                ax2.text(0.5, 0.5, 'no exo scores', ha='center', va='center'); ax2.axis('off')
            attn_ax_idx = 3
        else:
            axes[1].axis('off')

        # (3) Ego->Exo cross-attention averaged over heads.
        ax_attn = axes[attn_ax_idx-1]
        if attn is not None:
            A_full = _to_np(attn[b].mean(0))                 # (Tq_pad, Tk_pad)
            # Use the valid ego coarse length for the row count.
            if hasattr(model.sampler_ego, 'last_mask_coarse') and model.sampler_ego.last_mask_coarse is not None:
                Lq = int(model.sampler_ego.last_mask_coarse[b].float().sum().item())
            else:
                Lq = A_full.shape[0]
            # Use the sampled exo key count for the column count.
            Lk = A_full.shape[1]
            if hasattr(model, 'sampler_exo') and getattr(model.sampler_exo, 'last_index_map', None) is not None:
                idx_np = model.sampler_exo.last_index_map[b].detach().cpu().numpy()
                # Pad is usually -1; otherwise fall back to repeated tail values.
                if (idx_np < 0).any():
                    Lk = int((idx_np >= 0).sum())
                else:
                    tail = idx_np[-1] if len(idx_np) else -1
                    nz = np.where(idx_np != tail)[0]
                    Lk = int(nz[-1] + 1) if len(nz) else A_full.shape[1]
            # Crop to the valid attention region.
            A = A_full[:Lq, :Lk]
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            # Interpolate to a fixed display size for easier comparison.
            if A.size == 0:
                A_fixed = np.zeros((100, 100), dtype=np.float32)
            else:
                At = torch.from_numpy(A).unsqueeze(0).unsqueeze(0).float()      # [1,1,Lq,Lk]
                Af = F.interpolate(At, size=(100, 100),
                                   mode='bilinear', align_corners=False)[0,0]   # [H,W]
                # Restore row-wise normalization after interpolation.
                row_sum = Af.sum(dim=1, keepdim=True).clamp_min(1e-6)
                Af = Af / row_sum
                A_fixed = Af.clamp_min(0.0).cpu().numpy()
            # Keep a fixed color range across samples when possible.
            attn_vmax = 0.02
            vmin = 0.0
            vmax = attn_vmax if attn_vmax and attn_vmax > 0 else float(np.percentile(A_fixed, 99.5))
            im = ax_attn.imshow(
                A_fixed, origin='lower', aspect='auto',
                extent=[0.0, 1.0, 0.0, 1.0],  # Normalized axes.
                interpolation='nearest',
                vmin=vmin, vmax=vmax, cmap="YlOrRd"
            )
            ax_attn.set_xlim(0.0, 1.0); ax_attn.set_ylim(0.0, 1.0)
            ax_attn.set_title(f'Ego→Exo Cross-Attn (size={A.shape})')
            ax_attn.set_xlabel('normalized exo (0–1)')
            ax_attn.set_ylabel('normalized ego (0–1)')
            # ax_attn.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            # ax_attn.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            fig.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.02)
        else:
            ax_attn.text(0.5, 0.5, 'no cross-attn cached', ha='center', va='center')
            ax_attn.axis('off')

        fig.tight_layout()
        out_path = os.path.join(out_dir, f'{safe_key}_sampler_logits.png')
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        if filter_keys is not None:
            filter_keys.discard(raw_key)   # Mark the key as done to avoid duplicates.
        saved += 1
    return saved

def _viz_normalize(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Rescale curves for visualization only.

    mode: 'none' | 'z' | 'tanh' | 'minmax'
    """
    if mode == 'none':
        return x.detach()
    x = x.detach()
    if mode == 'z':
        mu = x.mean()
        sd = x.std().clamp_min(1e-6)
        return (x - mu) / sd
    if mode == 'tanh':
        scale = x.abs().mean().clamp_min(1e-6) * 3.0
        return torch.tanh(x / scale)
    if mode == 'minmax':
        mn, mx = x.min(), x.max()
        if (mx - mn) < 1e-6:
            return torch.zeros_like(x)
        return (x - mn) / (mx - mn)
    return x
