'''
Author: jacklee && lx_jacklee@163.com
Date: 2025-06-24 20:14:37
LastEditors: jacklee && lx_jacklee@163.com
LastEditTime: 2025-12-10 16:52:38
FilePath: \SAVAX\densevid_eval3\eval_imitation.py
Description: 

Copyright (c) 2025 by jacklee, All Rights Reserved. 
'''
# from densevid_eval3.evaluate2018_orig import main as eval2018

import json
# sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from collections import defaultdict

from sklearn.metrics import roc_auc_score, average_precision_score
Set=set
import numpy as np
def eval_imitation(json_path, reference, topN=1000, w_correct=0.2,
             phase='val',                    # 'val' or 'test'
             num_thresh=100,                 # Number of sampled thresholds
             thresh_sampling='quantile',     # 'quantile' or 'uniform'
             fix_fine_thresh=True,           # Whether to share one fine-grained threshold across all tIoUs
             global_fine_thresh=None,        # Test-time threshold; keep None during validation to search automatically
             tiou_for_thresh=None,            # tIoU used to pick the global threshold during validation (default: tious[0])
             global_overall_thresh=None
            ):
    args = type('args', (object,), {})()
    args.submission = json_path
    args.max_proposals_per_video = topN
    args.tious = [0.3,0.5,0.7,0.9]
    args.verbose = False

    args.references = reference
    args.w_correct = w_correct
    args.phase = phase
    args.num_thresh = num_thresh
    args.thresh_sampling = thresh_sampling
    args.fix_fine_thresh = fix_fine_thresh
    args.global_fine_thresh = global_fine_thresh
    args.global_overall_thresh = global_overall_thresh
    args.tiou_for_thresh = tiou_for_thresh
    eval_func = main
    score = eval_func(args)
    return score

# --------------------------------------------------------
# evaluation scripts for dense video captioning, support python 3
# Modified from https://github.com/ranjaykrishna/densevid_eval/tree/deba7d7e83012b218a4df888f6c971e21cfeea33
# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------



def compute_auc_from_curve(fpr, tpr):
    """
    Compute ROC AUC from FPR/TPR points with the trapezoidal rule.

    Args:
        fpr: array-like of shape (N,), false positive rate in [0, 1].
        tpr: array-like of shape (N,), true positive rate in [0, 1].

    Returns:
        float: Area under the ROC curve.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    
    # Sort to ensure FPR is monotonically increasing.
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    # Compute the area with trapezoidal integration.
    auc = np.trapz(tpr, fpr)
    return auc

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class ANETcaptions(object):
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000, 
                 prediction_fields=PREDICTION_FIELDS, verbose=False, **kargs):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        
        
        self.w_correct = kargs.get('w_correct', 0.5)
        self.phase = kargs.get('phase', 'val')  # 'val' or 'test'
        self.num_thresh = kargs.get('num_thresh', 100)
        self.thresh_sampling = kargs.get('thresh_sampling', 'quantile')  # 'quantile' or 'uniform'
        self.fix_fine_thresh = kargs.get('fix_fine_thresh', True)
        self.global_fine_threshold = kargs.get('global_fine_threshold', None)
        self.global_overall_threshold = kargs.get('global_overall_threshold', None)
        self.tiou_for_thresh = kargs.get('tiou_for_thresh', None)
        self.tious = tious
        if self.tiou_for_thresh is None and len(self.tious) > 0:
            self.tiou_for_thresh = self.tious[0]
            
            
        self.verbose = verbose
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.ground_truths_keys = [vid for gt in self.ground_truths for vid in gt ]
        eval_video_num = len(set(self.ground_truths_keys) & set(self.prediction.keys()))
        assert eval_video_num > 0
        print('available video number', eval_video_num)

        # Set up scorers, if not verbose, we only use the one we're
    def _make_thresholds(self, scores):
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]
        if scores.size == 0:
            return np.array([0.5], dtype=float)
        if self.thresh_sampling == 'quantile':
            qs = np.linspace(0.0, 1.0, max(2, self.num_thresh))
            # For numpy <1.22, np.percentile(scores, qs * 100) also works.
            th = np.quantile(scores, qs)
        else:
            th = np.linspace(scores.min(), scores.max(), max(2, self.num_thresh))
        th = np.unique(th)
        return th
    
    # =========================
    # COCO-STYLE INSERT ①
    # Compute COCO-style PR and AP: apply a monotonic envelope, then sample 101 recall points.
    # =========================
    def _coco_pr_ap(self, scores, is_tp, npos, recall_points=101, use_envelope=True):
        import numpy as np
        scores = np.asarray(scores, dtype=np.float64)
        is_tp  = np.asarray(is_tp,  dtype=np.int32)
        if npos <= 0:
            # No positives: following COCO convention, set AP to NaN and return a default PR point.
            return [1.0], [0.0], float('nan')

        if scores.size == 0:
            # No predictions: precision and recall stay at 0, so AP is 0.
            return [0.0], [0.0], 0.0

        # Stable descending sort by score.
        order = np.argsort(-scores, kind='mergesort')
        is_tp_sorted = is_tp[order]

        tp_cum = np.cumsum(is_tp_sorted)
        fp_cum = np.cumsum(1 - is_tp_sorted)

        precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
        recall    = tp_cum / float(npos)

        # Monotonic envelope: sweep from right to left and keep the suffix max of precision.
        if use_envelope:
            for i in range(precision.size - 2, -1, -1):
                precision[i] = max(precision[i], precision[i + 1])

        # Interpolated AP with 101 uniform recall samples.
        # For each recall r, take the maximum precision with recall >= r.
        rs = np.linspace(0.0, 1.0, recall_points)
        ps = []
        j = 0
        for r in rs:
            while j < recall.size and recall[j] < r:
                j += 1
            p_r = precision[j] if j < precision.size else 0.0
            ps.append(p_r)
        ap = float(np.mean(ps))

        # Return interpolated AP and the PR curve sampled at 101 points.
        return ps, rs.tolist(), ap


    # =========================
    # COCO-STYLE INSERT ②
    # Collect COCO-style bookkeeping for fine mimic at one tIoU:
    # - each prediction can match at most one GT
    # - score correct / error as separate classes
    #   correct: score = s
    #   error  : score = 1 - s (equivalent to sorting by ascending s)
    # - npos is the GT count for each class, including unmatched GTs
    # =========================
    def _collect_fine_coco_records(self, tiou):
        det_scores_c, det_tp_c = [], []
        det_scores_e, det_tp_e = [], []
        npos_c, npos_e = 0, 0
        def _eval_one_class(preds, ref_timestamps, gt_mask, score_fn):
            # gt_mask: np.bool_ array, True when the ref belongs to the current GT class.
            matched = np.zeros(len(ref_timestamps), dtype=bool)
            scores = np.array([float(score_fn(p)) for p in preds], dtype=float)
            order = np.argsort(-scores, kind="mergesort")  # Descending scores within this class.
            out_scores, out_tp = [], []
            for idx in order:
                pred = preds[idx]
                s = float(score_fn(pred))
                best_ref_i, best_iou = -1, 0.0
                for ref_i, ref_ts in enumerate(ref_timestamps):
                    if not gt_mask[ref_i] or matched[ref_i]:
                        continue
                    iou = self.iou(pred["timestamp"], ref_ts)
                    if iou > tiou and iou > best_iou:
                        best_iou, best_ref_i = iou, ref_i
                if best_ref_i >= 0:
                    matched[best_ref_i] = True
                    out_scores.append(s); out_tp.append(1)
                else:
                    out_scores.append(s); out_tp.append(0)
            return out_scores, out_tp
        gt_vid_ids = self.get_gt_vid_ids()
        for vid_id in gt_vid_ids:
            if vid_id not in self.prediction:
                continue
            preds = self.prediction.get(vid_id, [])
            # Only use the first GT entry containing this vid to avoid duplicate evaluation/counting.
            refs = None
            for gt in self.ground_truths:
                if vid_id in gt:
                    refs = gt[vid_id]
                    break
            if refs is None:
                continue
            ref_ts = refs["timestamps"]
            num_refs = len(ref_ts)
            # GT class labels.
            is_err_gt = np.zeros(num_refs, dtype=bool)
            for i in range(num_refs):
                sent = refs["sentences"][i]
                is_err_gt[i] = (("FALSE" in vid_id.upper()) and sent.startswith("F"))
            npos_e += int(is_err_gt.sum())
            npos_c += int((~is_err_gt).sum())
            # Larger s means more correct; for the error class we use 1 - s.
            score_c = lambda p: float(p.get("mimic_fine_score", 0.0))
            score_e = lambda p: 1.0 - float(p.get("mimic_fine_score", 0.0))
            # Match each class independently.
            sc, tc = _eval_one_class(preds, ref_ts, gt_mask=(~is_err_gt), score_fn=score_c)
            se, te = _eval_one_class(preds, ref_ts, gt_mask=( is_err_gt), score_fn=score_e)
            det_scores_c.extend(sc); det_tp_c.extend(tc)
            det_scores_e.extend(se); det_tp_e.extend(te)
        return (det_scores_c, det_tp_c, npos_c), (det_scores_e, det_tp_e, npos_e)


    # ===========================================
    # COCO-STYLE REPLACE: evaluate_fine_thresholdless
    #   - Switch to COCO-style AP/PR instead of sklearn.average_precision_score
    #   - Keep the existing keys: fine_auprc_* now stores COCO AP
    #   - Set fine_auroc_* to NaN because ROC is not directly comparable here
    # ===========================================
    def evaluate_fine_thresholdless(self, tiou):
        import numpy as np
        out = {}

        # Collect COCO bookkeeping.
        (sc_c, tp_c, P_c), (sc_e, tp_e, P_e) = self._collect_fine_coco_records(tiou)

        # Correct class.
        prec_c, rec_c, ap_c = self._coco_pr_ap(sc_c, tp_c, P_c)
        out['fine_pr_curve_correct'] = {'precision': prec_c, 'recall': rec_c}
        out['fine_auprc_correct'] = ap_c
        out['fine_auroc_correct'] = float('nan')  # Not meaningful in this detection setup.

        # Error class.
        prec_e, rec_e, ap_e = self._coco_pr_ap(sc_e, tp_e, P_e)
        out['fine_pr_curve_error'] = {'precision': prec_e, 'recall': rec_e}
        out['fine_auprc_error'] = ap_e
        out['fine_auroc_error'] = float('nan')    # Not meaningful in this detection setup.

        # Weighted mAP with the original key: w_correct * AP_c + (1 - w_correct) * AP_e.
        wc = float(getattr(self, 'w_correct', 0.5))
        if not (np.isnan(ap_c) or np.isnan(ap_e)):
            out['fine_map_weighted'] = wc * ap_c + (1.0 - wc) * ap_e
        else:
            out['fine_map_weighted'] = float('nan')
        return out

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission...")
        submission = json.load(open(prediction_filename))
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        for vid_id in submission['results']:
            results[vid_id] = submission['results'][vid_id][:self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = Set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(gt)
        if self.verbose:
           print("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids)))
        return gts

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)
    
    def _ensure_global_fine_threshold(self):
        if not self.fix_fine_thresh:
            return
        if self.global_fine_threshold is not None:
            return
        if self.phase == 'val':
            # Pick the global threshold on the specified tIoU (default: tious[0]).
            stats = self.evaluate_fine(self.tiou_for_thresh, force_threshold=None)
            self.global_fine_threshold = float(stats['best_threshold'])
        else:
            # If phase == test and no threshold is provided, fall back to 0.5.
            self.global_fine_threshold = 0.5
            
    # COCO-STYLE REPLACE
    def evaluate_fine(self, tiou, force_threshold=None):
        import numpy as np
        # 1) Collect COCO bookkeeping (each prediction matches at most one GT; npos includes unmatched GTs).
        (sc_c, tp_c, P_c), (sc_e, tp_e, P_e) = self._collect_fine_coco_records(tiou)
        sc_c = np.asarray(sc_c, dtype=np.float64); tp_c = np.asarray(tp_c, dtype=np.int32)
        sc_e = np.asarray(sc_e, dtype=np.float64); tp_e = np.asarray(tp_e, dtype=np.int32)
        # 2) Sort by descending score and precompute prefix sums for fast threshold queries.
        ord_c = np.argsort(-sc_c, kind='mergesort'); sc_c = sc_c[ord_c]
        tpc_cum = np.cumsum(tp_c[ord_c]); fpc_cum = np.cumsum(1 - tp_c[ord_c])
        ord_e = np.argsort(-sc_e, kind='mergesort'); sc_e = sc_e[ord_e]
        tpe_cum = np.cumsum(tp_e[ord_e]); fpe_cum = np.cumsum(1 - tp_e[ord_e])
        def _pr_at_thr(sc_desc, tp_cum, fp_cum, P, thr):
            # On descending scores, find the last position whose score is >= thr.
            n = sc_desc.size
            if n == 0 or P <= 0:
                return 0.0, 0.0
            pos_ge = n - np.searchsorted(sc_desc[::-1], thr, side='left')
            if pos_ge <= 0:
                return 0.0, 0.0
            idx = pos_ge - 1
            tp = float(tp_cum[idx]); fp = float(fp_cum[idx])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / float(P)
            return prec, rec
        def _f1(p, r):
            s = p + r
            return (2 * p * r / s) if s > 0 else 0.0
        # 3) Threshold candidates: prefer force_threshold, otherwise use model output scores only.
        if force_threshold is None:
            all_scores = np.array([float(pred.get('mimic_fine_score', 0.0))
                                for vid in self.prediction.values() for pred in vid],
                                dtype=np.float64)
            thresholds = self._make_thresholds(all_scores) if all_scores.size else np.array([0.5], dtype=float)
        else:
            thresholds = np.array([float(force_threshold)], dtype=float)
        wc = float(getattr(self, 'w_correct', 0.5))
        best_thr, best_wf1 = float(thresholds[0]), -1.0
        best_tuple = (0.0, 0.0, 0.0, 0.0)
        for thr in thresholds:
            pc, rc = _pr_at_thr(sc_c, tpc_cum, fpc_cum, P_c, thr)
            # `best_threshold` is always defined in the original s space (larger s means more correct).
            pe, re = _pr_at_thr(sc_e, tpe_cum, fpe_cum, P_e, thr)
            f1c, f1e = _f1(pc, rc), _f1(pe, re)
            wf1 = wc * f1c + (1.0 - wc) * f1e
            if wf1 > best_wf1:
                best_wf1 = wf1
                best_thr = float(thr)
                best_tuple = (pc, rc, pe, re)
        pc, rc, pe, re = best_tuple
        return {
            "best_threshold": float(best_thr),
            "best_weighted_f1": float(best_wf1),
            "precision_correct": float(pc),
            "recall_correct": float(rc),
            "precision_error": float(pe),
            "recall_error": float(re),
        }

    def evaluate_all_auc(self):
        gt_vid_ids = self.get_gt_vid_ids()
        y_true_correct, y_score = [], []
        for vid_id in gt_vid_ids:
            preds = self.prediction.get(vid_id, [])
            if not preds: 
                continue
            score = float(preds[0].get('mimic_overall_score', 0.0))  # Larger scores mean more "correct".
            is_correct = ('FALSE' not in vid_id.upper())
            y_true_correct.append(1 if is_correct else 0)
            y_score.append(score)

        y_true_correct = np.array(y_true_correct)
        y_score = np.array(y_score)
        y_true_error = 1 - y_true_correct  # Error is the positive class.

        # AUC / AUPRC without thresholding.
        out = {}
        if len(np.unique(y_true_correct)) > 1:
            out["all_auc_correct"]  = roc_auc_score(y_true_correct, y_score)
            out["all_auprc_correct"] = average_precision_score(y_true_correct, y_score)
        else:
            out["all_auc_correct"] = out["all_auprc_correct"] = float('nan')

        if len(np.unique(y_true_error)) > 1:
            # Error as the positive class: smaller scores mean more error, so use -y_score.
            out["all_auc_error"]   = roc_auc_score(y_true_error, -y_score)
            out["all_auprc_error"] = average_precision_score(y_true_error, -y_score)
        else:
            out["all_auc_error"] = out["all_auprc_error"] = float('nan')

        # Threshold-based metrics.
        def stats_at_thresh(th):
            # Correct as the positive class.
            pred_correct = (y_score >= th).astype(int)
            tp = np.sum((pred_correct==1)&(y_true_correct==1))
            fp = np.sum((pred_correct==1)&(y_true_correct==0))
            tn = np.sum((pred_correct==0)&(y_true_correct==0))
            fn = np.sum((pred_correct==0)&(y_true_correct==1))
            prec_c = tp / max(1, tp+fp)
            rec_c  = tp / max(1, tp+fn)
            f1_c   = 2*prec_c*rec_c / max(1e-12, (prec_c+rec_c))
            acc    = (tp+tn)/max(1, tp+tn+fp+fn)

            # Error as the positive class: equivalent to predicting "error" when y_score < th.
            pred_error = (y_score < th).astype(int)
            tp_e = np.sum((pred_error==1)&(y_true_error==1))
            fp_e = np.sum((pred_error==1)&(y_true_error==0))
            fn_e = np.sum((pred_error==0)&(y_true_error==1))
            prec_e = tp_e / max(1, tp_e+fp_e)
            rec_e  = tp_e / max(1, tp_e+fn_e)
            f1_e   = 2*prec_e*rec_e / max(1e-12, (prec_e+rec_e))
            return {
                "acc": acc,
                "prec_correct": prec_c, "recall_correct": rec_c, "f1_correct": f1_c,
                "prec_error": prec_e,   "recall_error":  rec_e,  "f1_error":  f1_e,
                "TP_e": int(tp_e), "FP_e": int(fp_e), "FN_e": int(fn_e)
            }

        if self.global_overall_threshold is not None:
            th = float(self.global_overall_threshold)
            s = stats_at_thresh(th)
            out.update({
                "all_best_threshold_correct_by_acc": th,      # Keep the existing metric convention.
                "all_accuracy_correct": s["acc"],
                "all_precision_correct": s["prec_correct"],
                "all_recall_correct": s["recall_correct"],
                "all_f1_correct": s["f1_correct"],
                # Important: metrics with error as the positive class.
                "threshold_error": th,                        # Backward compatible with older training scripts.
                "all_best_threshold_error_by_f1": th,         # Backward compatible alias.
                "all_threshold_error": th,
                "all_precision_error": s["prec_error"],
                "all_recall_error": s["recall_error"],
                "all_f1_error": s["f1_error"],
            })
            out["num_samples"] = int(len(y_true_correct))
            return out

        # Threshold search: if error discovery matters more, optimize for maximum f1_error.
        thresholds = self._make_thresholds(y_score)
        best_err = {"f1_error": -1.0}
        best_cor = {"acc": -1.0}
        for th in thresholds:
            s = stats_at_thresh(th)
            if s["f1_error"] > best_err["f1_error"]:
                best_err = dict(s, th=float(th))
            if s["acc"] > best_cor["acc"]:
                best_cor = dict(s, th=float(th))

        # Return both optimal thresholds for backward compatibility and error-focused evaluation.
        out.update({
            "all_best_threshold_correct_by_acc": best_cor["th"],
            "all_accuracy_correct": best_cor["acc"],
            "all_precision_correct": best_cor["prec_correct"],
            "all_recall_correct": best_cor["recall_correct"],
            "all_f1_correct": best_cor["f1_correct"],

            "threshold_error": best_err["th"],                # Backward compatible with older training scripts.
            "all_best_threshold_error_by_f1": best_err["th"],
            "all_threshold_error": best_err["th"],            # Backward compatible with the fixed-threshold branch.
            "all_precision_error": best_err["prec_error"],
            "all_recall_error": best_err["recall_error"],
            "all_f1_error": best_err["f1_error"],
        })
        out["num_samples"] = int(len(y_true_correct))
        return out


    def evaluate(self):
        aggregator = {}
        self.scores = defaultdict(list)

        # Overall AUC and fine-grained evaluation.
        self.scores['iou_thres'] = self.tious

        all_score = self.evaluate_all_auc()
        self.scores.update(all_score)
        for item in self.scores:
            if type(self.scores[item])!= list:
                self.scores[item] = [self.scores[item]]  

        # Fine-grained threshold policy for validation/test.
        if self.fix_fine_thresh:
            self._ensure_global_fine_threshold()
            # Record the global threshold once.
            self.scores['thres_fine_global'] = [self.global_fine_threshold]

        for tiou in self.tious:
            # Use either a fixed threshold or search for one.
            if self.fix_fine_thresh:
                mimic_score = self.evaluate_fine(tiou, force_threshold=self.global_fine_threshold)
            else:
                mimic_score = self.evaluate_fine(tiou, force_threshold=None)

            self.scores['weighted_f1_fine'].append(mimic_score['best_weighted_f1'])
            self.scores['precision_fine_correct'].append(mimic_score['precision_correct'])
            self.scores['recall_fine_correct'].append(mimic_score['recall_correct'])
            self.scores['precision_fine_error'].append(mimic_score['precision_error'])
            self.scores['recall_fine_error'].append(mimic_score['recall_error'])
            self.scores['thres_fine'].append(mimic_score['best_threshold'])

            # Threshold-free fine-grained metrics.
            tl = self.evaluate_fine_thresholdless(tiou)
            self.scores['fine_auprc_correct'].append(tl['fine_auprc_correct'])
            self.scores['fine_auprc_error'].append(tl['fine_auprc_error'])
            self.scores['fine_pr_curve_correct'].append(tl['fine_pr_curve_correct'])
            self.scores['fine_pr_curve_error'].append(tl['fine_pr_curve_error'])
    
def main(args):
    args.verbose = False #True
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             verbose=args.verbose,
                             w_correct=args.w_correct,
                             phase=args.phase,
                             num_thresh=args.num_thresh,
                             thresh_sampling=args.thresh_sampling,
                             fix_fine_thresh=args.fix_fine_thresh,
                             global_fine_threshold=args.global_fine_thresh,
                             global_overall_threshold=args.global_overall_thresh,
                             tiou_for_thresh=args.tiou_for_thresh)
    evaluator.evaluate()
    # evaluator.scores['tiou'] = args.tious
    return evaluator.scores
