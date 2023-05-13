import torch
import numpy as np
import os.path as osp

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, roc_curve, f1_score, accuracy_score, precision_score, recall_score

def inference(dataloader, model, args, device, rec_auc_only=True):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)

        gt = dataloader.dataset.ground_truths
        dataset = args.dataset.lower()

        if args.inference:
            video_list = dataloader.dataset.video_list
            result_dict = dict()

        for i, (video, macro) in enumerate(dataloader):
            video = video.to(device)
            video = video.permute(0, 2, 1, 3)

            macro = macro.to(device)

            outputs = model(video, macro)

            # >> parse outputs
            logits = outputs['video_scores']

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            if args.inference:
                video_id = video_list[i]
                result_dict[video_id] = logits.cpu().detach().numpy()

            sig = logits
            pred = torch.cat((pred, sig))

        if args.inference:
            out_dir = f'output/{dataset}'
            import pickle
            with open(osp.join(out_dir, f'{dataset}_taskaware_results.pickle'), 'wb') as fout:
                pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)

        if rec_auc_only:
            return rec_auc

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)

        f1 = f1_score(gt, np.rint(pred))
        f1_macro = f1_score(gt, np.rint(pred), average="macro")
        acc = accuracy_score(gt, np.rint(pred))
        prec = precision_score(gt, np.rint(pred))
        recall = recall_score(gt, np.rint(pred))
        ap = average_precision_score(gt, pred)

        score = {"rec_auc": rec_auc,
                 "pr_auc": pr_auc,
                 "f1": f1,
                 "f1_macro": f1_macro,
                 "accuracy": acc,
                 "precision": prec,
                 "recall": recall,
                 "average_precision": ap}

        return score
