import numpy as np

def compute_iou(pred_seg, gt_seg):
    """
    Computes the Intersection over Union (IoU) between two temporal segments.
    
    Args:
        pred_seg (tuple): (start_time, end_time) of the predicted segment.
        gt_seg (tuple): (start_time, end_time) of the ground truth segment.
    
    Returns:
        iou (float): Intersection over Union value.
    """
    start_pred, end_pred = pred_seg
    start_gt, end_gt = gt_seg
    # Calculate intersection
    inter_start = max(start_pred, start_gt)
    inter_end = min(end_pred, end_gt)
    intersection = max(0.0, inter_end - inter_start)
    # Calculate union
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def compute_average_precision(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision (AP) for a single class.
    
    Args:
        predictions (list of dict): Each dict should have:
            - 'video_id'
            - 'start_time'
            - 'end_time'
            - 'confidence'
        ground_truths (dict): A dict where keys are video_ids and values are 
            lists of ground truth segments for that class. Each ground truth segment 
            is a tuple: (start_time, end_time).
        iou_threshold (float): The IoU threshold to consider a prediction as True Positive.
        
    Returns:
        AP (float): Average precision for the class.
    """
    # Sort predictions by confidence in descending order
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    # Initialize true positive (TP) and false positive (FP) lists
    tp = [0] * len(predictions)
    fp = [0] * len(predictions)
    # Keep track of which ground truth segments have been detected per video
    detected = {video_id: [False] * len(ground_truths.get(video_id, []))
                for video_id in ground_truths}
    for i, pred in enumerate(predictions):
        vid = pred['video_id']
        pred_seg = (pred['start_time'], pred['end_time'])
        max_iou = 0.0
        max_j = -1
        if vid not in ground_truths or len(ground_truths[vid]) == 0:
            fp[i] = 1
            continue
        for j, gt_seg in enumerate(ground_truths[vid]):
            iou = compute_iou(pred_seg, gt_seg)
            if iou > max_iou:
                max_iou = iou
                max_j = j
        if max_iou >= iou_threshold:
            if not detected[vid][max_j]:
                tp[i] = 1  # True positive
                detected[vid][max_j] = True
            else:
                fp[i] = 1  # Duplicate detection is a false positive
        else:
            fp[i] = 1  # IoU too low
    # Cumulative sums for precision and recall calculation
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    total_gt = sum([len(ground_truths[vid]) for vid in ground_truths])
    recalls = tp_cumsum / (total_gt + 1e-8)
    # Calculate AP by summing over recall changes
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap

def compute_map(all_predictions, all_ground_truths, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) across all classes.
    
    Args:
        all_predictions (list of dict): All predicted segments from the model.
        all_ground_truths (dict): Ground truth segments in the format:
            {class_label: {video_id: [(start_time, end_time), ...], ...}, ...}
        iou_threshold (float): IoU threshold for determining true positives.
    
    Returns:
        mAP (float): Mean Average Precision across classes.
        ap_dict (dict): AP for each class.
    """
    class_labels = [cls for cls in all_ground_truths.keys() if cls != 0]  # Exclude background
    ap_dict = {}
    for cls in class_labels:
        cls_preds = [pred for pred in all_predictions if pred['label'] == cls]
        cls_gts = all_ground_truths[cls]
        ap = compute_average_precision(cls_preds, cls_gts, iou_threshold=iou_threshold)
        ap_dict[cls] = ap
    mAP = np.mean(list(ap_dict.values())) if len(ap_dict) > 0 else 0.0
    return mAP, ap_dict

def post_process_predictions(predictions, min_gap=0.0):
    """
    Group sliding window predictions into continuous segments for each video.

    Args:
        predictions (list of dict): Each dict should have keys:
            'video_id', 'window_start_time', 'window_end_time', 'label', 'action',
            and optionally 'confidence'.
        min_gap (float): Maximum gap (in seconds) allowed between consecutive windows for merging.
    
    Returns:
        merged_segments (dict): Keys are video IDs, and values are lists of segments.
            Each segment is a dict with keys: 'video_id', 'start_time', 'end_time',
            'label', 'action', 'confidence' (average confidence).
    """
    # Group predictions by video_id
    video_preds = {}
    for pred in predictions:
        vid = pred['video_id']
        video_preds.setdefault(vid, []).append(pred)
    
    merged_segments = {}
    for vid, preds in video_preds.items():
        preds = sorted(preds, key=lambda x: x['window_start_time'])
        segments = []
        current_seg = None
        for pred in preds:
            if pred['label'] == 0:
                continue
            if current_seg is None:
                current_seg = {
                    'video_id': vid,
                    'start_time': pred['window_start_time'],
                    'end_time': pred['window_end_time'],
                    'label': pred['label'],
                    'action': pred['action'],
                    'confidences': [pred.get('confidence', 1.0)]
                }
            else:
                if (pred['label'] == current_seg['label'] and 
                    pred['window_start_time'] - current_seg['end_time'] <= min_gap):
                    current_seg['end_time'] = max(current_seg['end_time'], pred['window_end_time'])
                    current_seg['confidences'].append(pred.get('confidence', 1.0))
                else:
                    current_seg['confidence'] = sum(current_seg['confidences']) / len(current_seg['confidences'])
                    del current_seg['confidences']
                    segments.append(current_seg)
                    current_seg = {
                        'video_id': vid,
                        'start_time': pred['window_start_time'],
                        'end_time': pred['window_end_time'],
                        'label': pred['label'],
                        'action': pred['action'],
                        'confidences': [pred.get('confidence', 1.0)]
                    }
        if current_seg is not None:
            current_seg['confidence'] = sum(current_seg['confidences']) / len(current_seg['confidences'])
            del current_seg['confidences']
            segments.append(current_seg)
        merged_segments[vid] = segments
    return merged_segments
