import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
from shutil import rmtree
from tqdm import tqdm
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from pytorch_fid import fid_score
from sklearn.metrics import accuracy_score


class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=0.001):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = (2 * numer + self.smooth) / (denor + self.smooth)
        return loss
    
_dice_loss_ = SoftDiceLossV1()

def dice_loss(masks):
    shape = masks.shape
    idxs = np.arange(shape[1])

    masks = (masks - masks.min(dim=1, keepdim=True)[0] + 1e-4)/(1e-4 + masks.max(dim=1, keepdim=True)[0] - masks.min(dim=1, keepdim=True)[0])
    gt_masks = masks.clone().detach()
    gt_masks[gt_masks >= 0.5] = 1.0
    gt_masks[gt_masks < 0.5] = 0.0
    
    loss = 0
    for i in idxs:
        _idxs_ = list(np.arange(shape[1]))
        del _idxs_[i]
        loss += _dice_loss_(masks[:, _idxs_, ...].reshape(-1, shape[2], shape[3], shape[4]),
            gt_masks[:, i, ...].unsqueeze(1).repeat(1, len(idxs) -1, 1, 1, 1).reshape(-1, shape[2], shape[3], shape[4]))

    return loss*1.0/len(idxs)



@torch.no_grad()
def ReliableReasoningIndex(x, y):
    """
    code conversion from: https://github.com/google-research/google-research/blob/c3bef5045a2d4777a80a1fb333d31e03474222fb/slot_attention/utils.py#L26
    
    Huber loss for sets, matching elements with the Hungarian algorithm.
    This loss is used as reconstruction loss in the paper 'Deep Set Prediction
    Networks' https://arxiv.org/abs/1906.06565, see Eq. 2. For each element in the
    batches we wish to compute min_{pi} ||y_i - x_{pi(i)}||^2 where pi is a
    permutation of the set elements. We first compute the pairwise distances
    between each point in both sets and then match the elements using the scipy
    implementation of the Hungarian algorithm. This is applied for every set in
    the two batches. Note that if the number of points does not match, some of the
    elements will not be matched. As distance function we use the Huber loss.
    Args:
    x: Batch of sets of size [batch_size, n_points, dim_points]. Each set in the
        batch contains n_points many points, each represented as a vector of
        dimension dim_points.
    y: Batch of sets of size [batch_size, n_points, dim_points].
    Returns:
    Average distance between all sets in the two batches.
    """

    # adjust shape for x and y
    ns = x.shape[1]
    x = x.unsqueeze(1).repeat(1, ns, 1, 1)
    y = y.unsqueeze(2).repeat(1, 1, ns, 1)
    
    pairwise_cost = F.huber_loss(x, y, reduction='none')
    # pairwise_cost = F.binary_cross_entropy(x, y, reduction='none')
    pairwise_cost = pairwise_cost.mean(-1)

    indices = np.array(list(map(linear_sum_assignment, pairwise_cost.clone().detach().cpu().numpy())))
    transposed_indices = np.transpose(indices, axes=(0, 2, 1))

    transposed_indices = torch.tensor(transposed_indices).to(x.device)
    pos_h = torch.tensor(transposed_indices)[:,:,0]
    pos_w = torch.tensor(transposed_indices)[:,:,1]
    batch_enumeration = torch.arange(x.shape[0])
    
    batch_enumeration = batch_enumeration.unsqueeze(1)
    actual_costs = pairwise_cost[batch_enumeration, pos_h, pos_w]
    
    return actual_costs.sum(1).mean()




@torch.no_grad()
def calculate_fid(loader, model, batch_size=16, num_batches=100, fid_dir='./tmp/CLEVR/FID' ):
    torch.cuda.empty_cache()

    real_path = os.path.join(fid_dir, 'real')
    fake_path = os.path.join(fid_dir, 'fake')

    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    # remove any existing files used for fid calculation and recreate directories

    if len(os.listdir(real_path)) < 10 :
        rmtree(real_path, ignore_errors=True)
        os.makedirs(real_path)

        for batch_num, samples in tqdm(enumerate(loader), desc='calculating FID - saving reals'):
            real_batch = samples['image']
            for k, image in enumerate(real_batch.cpu()):
                filename = str(k + batch_num * batch_size)
                torchvision.utils.save_image(image, os.path.join(real_path, f'{filename}.png'))

    # generate a bunch of fake images in results / name / fid_fake

    rmtree(fake_path, ignore_errors=True)
    os.makedirs(fake_path)

    model.eval()
   
    for batch_num, samples in tqdm(enumerate(loader), desc='calculating FID - saving generated'):

        image = samples['image'].to(model.device)
        recon_combined, *_ = model(image)
       
        for j, image in enumerate(recon_combined.cpu()):
            torchvision.utils.save_image(image, os.path.join(fake_path, f'{str(j + batch_num * batch_size)}.png'))

    return fid_score.calculate_fid_given_paths(paths = [str(real_path), str(fake_path)], 
                                                dims = 2048, 
                                                device=0,
                                                batch_size= 256, 
                                                num_workers = 8)


@torch.no_grad()
def calculate_sfid(loader, model, batch_size=16, num_batches=100, fid_dir='./tmp/CLEVR/FID' ):
    torch.cuda.empty_cache()

    real_path = os.path.join(fid_dir, 'real')
    fake_path = os.path.join(fid_dir, 'fake')

    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    # remove any existing files used for fid calculation and recreate directories

    if len(os.listdir(real_path)) < 10 :
        rmtree(real_path, ignore_errors=True)
        os.makedirs(real_path)

        for batch_num, samples in tqdm(enumerate(loader), desc='calculating FID - saving reals'):
            real_batch = samples['image']
            for k, image in enumerate(real_batch.cpu()):
                filename = str(k + batch_num * batch_size)
                torchvision.utils.save_image(image, os.path.join(real_path, f'{filename}.png'))

    # generate a bunch of fake images in results / name / fid_fake

    rmtree(fake_path, ignore_errors=True)
    os.makedirs(fake_path)
    for i in range(model.num_slots):
        os.makedirs(os.path.join(fake_path, f'slots-{i}'), exist_ok=True)

    model.eval()
   
    for batch_num, samples in tqdm(enumerate(loader), desc='calculating FID - saving generated'):

        image = samples['image'].to(model.device)
        recon_combined, recons, masks, *_ = model(image)

        recons = recons* masks + (1 - masks)
        for i in range(model.num_slots):
            for j, image in enumerate(recons[:, i, ...].cpu()):
                torchvision.utils.save_image(image.permute(2, 0, 1), os.path.join(fake_path, f'slots-{i}', f'{str(j + batch_num * batch_size)}.png'))

    fid_list = [fid_score.calculate_fid_given_paths(paths = [str(real_path), os.path.join(str(fake_path), f'slots-{i}')], 
                                                dims = 2048, 
                                                device=0,
                                                batch_size= 256, 
                                                num_workers = 8) for i in range(model.num_slots)]
    return np.mean(fid_list)





@torch.no_grad()
def compositional_fid(loader, 
                        model,
                        ns=10,
                        device=0, 
                        batch_size=16, 
                        num_batches=100, 
                        sfid=True,
                        save_imgs=False,
                        fid_dir='./tmp/CLEVR/FID' ):

    torch.cuda.empty_cache()

    real_path = os.path.join(fid_dir, 'real')
    fake_path = os.path.join(fid_dir, 'fake')

    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    # remove any existing files used for fid calculation and recreate directories

    if (len(os.listdir(real_path)) < 10) and (not save_imgs):
        rmtree(real_path, ignore_errors=True)
        os.makedirs(real_path)

        for batch_num, samples in tqdm(enumerate(loader), desc='calculating FID - saving reals'):
            real_batch = samples['image']
            for k, image in enumerate(real_batch.cpu()):
                filename = str(k + batch_num * batch_size)
                torchvision.utils.save_image(image, os.path.join(real_path, f'{filename}.png'))

    model.eval()

    # Generate undead codes
    dictionary_codes = []
    for batch_num, samples in tqdm(enumerate(loader), desc='generating propmt prior'):
        if batch_num > 100: break

        image = samples['image'].to(model.device)
        recon_combined, recons, masks, slots, attns = model(image)
        dictionary_codes.append(cbidxs)

    dictionary_codes = torch.cat(dictionary_codes, 0)
    dictionary_codes = torch.unique(dictionary_codes).detach().cpu().numpy()

    print (f'Prior indxs: {dictionary_codes}=========*************************')
    # generate a bunch of fake images in results / name / fid_fake

    rmtree(fake_path, ignore_errors=True)
    os.makedirs(fake_path)

    if sfid:
        for i in range(ns):
            os.makedirs(os.path.join(fake_path, f'slots-{i}'), exist_ok=True)
   

   
    for batch_num, samples in tqdm(enumerate(loader), desc='calculating FID - saving generated'):

        image = samples['image'].to(model.device)
        recon_combined, recons, masks, slots, attns = model.object_composition(n_s=ns,
                                                        prior=dictionary_codes,
                                                        b=batch_size,
                                                        device=device)
        recons = recons* masks + (1 - masks)
        
        if save_imgs:
            pictures_path = os.path.join(fid_dir, 'pictures')
            os.makedirs(pictures_path, exist_ok=True)

            _, _, H, W = recon_combined.shape
            attns = recons.permute(0, 1, 4, 2, 3)
            recon_combined = recon_combined.expand(-1, 3, H, W).unsqueeze(dim=1)
            attns = attns.expand(-1, -1, 3, H, W)
            pictures = torch.cat((attns, recon_combined), dim=1).view(-1, 3, H, W)
            grid = torchvision.utils.make_grid(pictures, nrow=ns + 1, pad_value=0.2)[:, 2:-2, 2:-2]
            torchvision.utils.save_image(grid, os.path.join(pictures_path, f'{str(batch_num)}.png'))
            return 0, 0

        for j, image in enumerate(recon_combined.cpu()):
            torchvision.utils.save_image(image, os.path.join(fake_path, f'{str(j + batch_num * batch_size)}.png'))

        if sfid:
            for i in range(ns):
                for j, image in enumerate(recons[:, i, ...].cpu()):
                    torchvision.utils.save_image(image.permute(2, 0, 1), os.path.join(fake_path, f'slots-{i}', f'{str(j + batch_num * batch_size)}.png'))


    fidvalue = fid_score.calculate_fid_given_paths(paths = [str(real_path), str(fake_path)], 
                                                dims = 2048, 
                                                device=0,
                                                batch_size= 256, 
                                                num_workers = 8)

    if sfid:
        fid_list = [fid_score.calculate_fid_given_paths(paths = [str(real_path), os.path.join(str(fake_path), f'slots-{i}')], 
                                                dims = 2048, 
                                                device=0,
                                                batch_size= 256, 
                                                num_workers = 8) for i in range(ns)]
        sfidvalue = np.mean(fid_list)

        return fidvalue, sfidvalue
        
    return fidvalue


# set prediction evaluation metrics
@torch.no_grad()
def average_precision_clevr(pred, attributes, distance_threshold):
    """
    Base Code from: https://github.com/google-research/google-research/blob/c3bef5045a2d4777a80a1fb333d31e03474222fb/slot_attention/utils.py#L60

    Computes the average precision for CLEVR.
    This function computes the average precision of the predictions specifically
    for the CLEVR dataset. First, we sort the predictions of the model by
    confidence (highest confidence first). Then, for each prediction we check
    whether there was a corresponding object in the input image. A prediction is
    considered a true positive if the discrete features are predicted correctly
    and the predicted position is within a certain distance from the ground truth
    object.
    Args:
    pred: np.ndarray of shape [batch_size, num_elements, dimension] containing
        predictions. The last dimension is expected to be the confidence of the
        prediction.
    attributes: np.ndarray of shape [batch_size, num_elements, dimension] containing
        ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
    Returns:
    Average precision of the predictions.
    """
    # =============================================

    [batch_size, _, element_size] = attributes.shape
    [_, predicted_elements, _] = pred.shape

    def unsorted_id_to_image(detection_id, predicted_elements):
        """Find the index of the image from the unsorted detection index."""
        return int(detection_id // predicted_elements)

    flat_size = batch_size * predicted_elements
    flat_pred = np.reshape(pred, [flat_size, element_size])
    sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

    sorted_predictions = np.take_along_axis(
        flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
    idx_sorted_to_unsorted = np.take_along_axis(
        np.arange(flat_size), sort_idx, axis=0)

    def process_targets(target):
        """Unpacks the target into the CLEVR properties."""
        object_size = np.argmax(target[:2])
        shape = np.argmax(target[2:5])
        color = np.argmax(target[5:13])
        material = np.argmax(target[13:15])
        real_obj = target[15]
        coords = target[16:]
        return coords, object_size, material, shape, color, real_obj

    true_positives = np.zeros(sorted_predictions.shape[0])
    false_positives = np.zeros(sorted_predictions.shape[0])

    detection_set = set()

    for detection_id in range(sorted_predictions.shape[0]):
        # Extract the current prediction.
        current_pred = sorted_predictions[detection_id, :]
        # Find which image the prediction belongs to. Get the unsorted index from
        # the sorted one and then apply to unsorted_id_to_image function that undoes
        # the reshape.
        original_image_idx = unsorted_id_to_image(
            idx_sorted_to_unsorted[detection_id], predicted_elements)
        # Get the ground truth image.
        gt_image = attributes[original_image_idx, :, :]

        # Initialize the maximum distance and the id of the groud-truth object that
        # was found.
        best_distance = 10000
        best_id = None

        # Unpack the prediction by taking the argmax on the discrete attributes.
        (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
            _) = process_targets(current_pred)

        # Loop through all objects in the ground-truth image to check for hits.
        for target_object_id in range(gt_image.shape[0]):
            target_object = gt_image[target_object_id, :]
            # Unpack the targets taking the argmax on the discrete attributes.
            (target_coords, target_object_size, target_material, target_shape,
            target_color, target_real_obj) = process_targets(target_object)
            # Only consider real objects as matches.
            if target_real_obj:
            # For the match to be valid all attributes need to be correctly
                # predicted.
                pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
                target_attr = [
                    target_object_size, target_material, target_shape, target_color]
                match = pred_attr == target_attr
                if match:
                    # If a match was found, we check if the distance is below the
                    # specified threshold. Recall that we have rescaled the coordinates
                    # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
                    # `pred_coords`. To compare in the original scale, we thus need to
                    # multiply the distance values by 6 before applying the norm.
                    distance = np.linalg.norm((target_coords - pred_coords) * 6.)

                    # If this is the best match we've found so far we remember it.
                    if distance < best_distance:
                        best_distance = distance
                        best_id = target_object_id

        if best_distance < distance_threshold or distance_threshold == -1:
            # We have detected an object correctly within the distance confidence.
            # If this object was not detected before it's a true positive.
            if best_id is not None:
                if (original_image_idx, best_id) not in detection_set:
                    true_positives[detection_id] = 1
                    detection_set.add((original_image_idx, best_id))
                else:
                    false_positives[detection_id] = 1
            else:
                false_positives[detection_id] = 1
        else:
            false_positives[detection_id] = 1
    accumulated_fp = np.cumsum(false_positives)
    accumulated_tp = np.cumsum(true_positives)
    recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
    precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

    return compute_average_precision(
        np.array(precision_array, dtype=np.float32),
        np.array(recall_array, dtype=np.float32))



def compute_average_precision(precision, recall):
    """
    Base Code from: https://github.com/google-research/google-research/blob/c3bef5045a2d4777a80a1fb333d31e03474222fb/slot_attention/utils.py#L183
    Computation of the average precision from precision and recall arrays.
    """
    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
        i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
    return average_precision


@torch.no_grad()
def accuracy(loader, model, num_batches):
    model.eval()

    labels = []; predictions = []
    for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
        samples = next(iter(loader))

        image = samples['image'].to(model.device)
        labels = samples['label'].to(model.device)

        _, _, logits, *_ = model(image, 
                                epoch=0, 
                                batch=batch_num)

        predictions.append(torch.argmax(logits, 1))
        labels.append(labels)

    predictions = torch.cat(predictions, 0).cpu().numpy()
    labels = torch.cat(labels, 0).cpu().numpy()

    return accuracy_score(labels, predictions)
