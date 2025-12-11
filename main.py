import random
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


class PointerNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        
        # Encoder for driver features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        
        # Pointer output
        self.pointer = nn.Linear(hidden_dim, 1)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # x: [batch_size, num_drivers, num_features]
        batch_size, num_drivers, num_features = x.shape
        
        # Encode all drivers
        x_flat = x.view(-1, num_features)
        encoded = self.encoder(x_flat)  # [batch_size * num_drivers, hidden_dim]
        encoded = encoded.view(batch_size, num_drivers, self.hidden_dim)
        
        # Initialize decoder
        decoder_input = encoded.mean(dim=1)  # Use mean as initial input
        decoder_hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        decoder_cell = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Generate sequence of pointers
        pointers = []
        mask = torch.ones(batch_size, num_drivers, device=x.device)
        
        for i in range(num_drivers):
            # Decoder step
            decoder_hidden, decoder_cell = self.decoder_lstm(decoder_input, (decoder_hidden, decoder_cell))
            
            # Attention over remaining drivers
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, num_drivers, -1)
            attention_input = torch.cat([encoded, decoder_hidden_expanded], dim=2)
            attention_weights = torch.tanh(self.attention(attention_input))
            scores = self.attention_score(attention_weights).squeeze(-1)
            
            # Mask out already selected drivers
            scores = scores.masked_fill(mask == 0, float('-inf'))
            probs = torch.softmax(scores, dim=1)
            
            # Sample or take argmax (during inference)
            if self.training:
                selected = torch.multinomial(probs, 1).squeeze(-1)
            else:
                selected = torch.argmax(probs, dim=1)
            
            pointers.append(selected)
            
            # Update mask and decoder input
            mask.scatter_(1, selected.unsqueeze(1), 0)
            decoder_input = encoded.gather(1, selected.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)).squeeze(1)
        
        return torch.stack(pointers, dim=1)  # [batch_size, num_drivers]


class PositionPredictionNetwork(nn.Module):
    # Neural network for predicting race finishing positions with conditional DNF modeling
    
    def __init__(self, input_dim, hidden_layers=(128, 64), dropout=0.2, activation='relu', use_pointer=False, use_conditional=False):
        super().__init__()
        
        self.use_pointer = use_pointer
        self.use_conditional = use_conditional
        
        if use_pointer:
            self.pointer_net = PointerNetwork(input_dim, hidden_dim=128, dropout=dropout)
        else:
            activation_map = {
                'relu': nn.ReLU(),
                'leaky_relu': nn.LeakyReLU(),
                'elu': nn.ELU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid()
            }
            act_fn = activation_map.get(activation, nn.ReLU())
            
            # Shared encoder for all drivers
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_layers:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    type(act_fn)(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            
            self.encoder = nn.Sequential(*layers)
            
            if use_conditional:
                # Conditional modeling: predict DNF first, then positions conditioned on DNF
                self.dnf_head = nn.Linear(prev_dim, 1)  # Binary DNF prediction
                # Position prediction takes DNF status as additional input
                self.position_head = nn.Linear(prev_dim + 1, 21)
            else:
                # Standard position prediction
                self.position_head = nn.Linear(prev_dim, 21)
        
    def forward(self, x):
        # x: [batch_size, num_drivers (20), num_features]
        if self.use_pointer:
            # Return pointer indices directly
            return self.pointer_net(x)
        else:
            batch_size, num_drivers, num_features = x.shape
            x_flat = x.view(-1, num_features)
            encoded = self.encoder(x_flat)  # [batch_size * num_drivers, hidden_dim]
            
            if self.use_conditional:
                # Predict DNF probabilities first
                dnf_logits = self.dnf_head(encoded)  # [batch_size * num_drivers, 1]
                dnf_probs = torch.sigmoid(dnf_logits)
                
                # Concatenate DNF probs with encoded features for position prediction
                position_input = torch.cat([encoded, dnf_probs], dim=1)  # [batch_size * num_drivers, hidden_dim + 1]
                position_logits = self.position_head(position_input)
                
                # Reshape outputs
                dnf_probs = dnf_probs.view(batch_size, num_drivers, 1)
                position_logits = position_logits.view(batch_size, num_drivers, 21)
                
                return position_logits, dnf_probs
            else:
                # Standard prediction
                logits = self.position_head(encoded)
                return logits.view(batch_size, num_drivers, 21)


class F1PositionPredictionPipeline:
    # End-to-end pipeline for F1 race position prediction
    
    def __init__(self, data_file, device='cpu', seed=339, use_pointer=False, use_conditional=False):
        self.data_file = data_file
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.use_pointer = use_pointer
        self.use_conditional = use_conditional
        
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        # history of flattened parameter vectors (one per epoch)
        self.theta_history = []

        

    
        

        

    def load_data(self):
        # Load preprocessed race data
        print("="*80)
        print("Loading Preprocessed Data")
        print("="*80)
        
        data = np.load(self.data_file, allow_pickle=True)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.feature_names = data['feature_names']
        self.race_metadata = data['race_metadata']
        
        print(f"Training set: {self.X_train.shape[0]} races")
        print(f"Test set: {self.X_test.shape[0]} races")
        print(f"Features per driver: {self.X_train.shape[2]}\n")

    def _flatten_params(self, model: nn.Module) -> np.ndarray:
        
        with torch.no_grad():
            return torch.cat([p.detach().cpu().flatten() for p in model.parameters()]).numpy()
    
    def create_mask(self, y):
        # Create mask for valid drivers (position <= 21)
        return (y <= 21).astype(np.float32)
    
    def hungarian_matching(self, pred_probs, mask=None):
        # Use Hungarian algorithm to assign unique positions to drivers
        batch_size, num_drivers, num_positions = pred_probs.shape
        assignments = np.zeros((batch_size, num_drivers), dtype=np.int32)
        
        for b in range(batch_size):
            cost_matrix = -pred_probs[b].cpu().detach().numpy()
            
            if mask is not None:
                valid_drivers = mask[b].cpu().numpy().astype(bool)
                cost_matrix[~valid_drivers] = 1e9
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments[b, row_ind] = col_ind + 1
        
        return assignments
    
    def adjust_positions_for_dnfs(self, assignments):
        # Adjust positions to account for DNFs - compress finishing positions when DNFs occur
        batch_size, num_drivers = assignments.shape
        adjusted_assignments = np.zeros_like(assignments)
        
        for b in range(batch_size):
            race_assignments = assignments[b]
            
            # Identify DNF drivers (position 21)
            dnf_mask = race_assignments == 21
            non_dnf_mask = ~dnf_mask
            
            # Get non-DNF drivers and their assigned positions
            non_dnf_indices = np.where(non_dnf_mask)[0]
            non_dnf_positions = race_assignments[non_dnf_indices]
            
            # Sort non-DNF drivers by their predicted positions
            sorted_order = np.argsort(non_dnf_positions)
            
            # Reassign positions: DNFs stay 21, others get 1, 2, 3, ... in order
            adjusted_race = np.full(num_drivers, 21, dtype=np.int32)
            for new_pos, idx_in_non_dnf in enumerate(sorted_order):
                driver_idx = non_dnf_indices[idx_in_non_dnf]
                adjusted_race[driver_idx] = new_pos + 1
            
            adjusted_assignments[b] = adjusted_race
        
        return adjusted_assignments
    
    def pointer_to_assignments(self, pointers):
        # Convert pointer network output to position assignments
        batch_size, num_drivers = pointers.shape
        assignments = np.zeros((batch_size, num_drivers), dtype=np.int32)
        
        for b in range(batch_size):
            pointer_sequence = pointers[b]  # [num_drivers] - driver indices in finishing order
            for position, driver_idx in enumerate(pointer_sequence):
                assignments[b, driver_idx] = position + 1  # positions start from 1
        
        return assignments
    
    def compute_position_accuracy(self, predictions, targets, mask=None, tolerance=0):
        # Compute accuracy of position predictions with tolerance
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        valid_mask = target_valid <= 21
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        within_tolerance = np.abs(pred_valid - target_valid) <= tolerance
        correct = within_tolerance.sum()
        total = len(target_valid)
        
        return correct / total if total > 0 else 0.0
    
    def compute_bin_accuracy(self, predictions, targets, mask=None):
        # Compute accuracy using position bins (Podium/Points/Midfield/Back)
        def position_to_bin(pos):
            if pos <= 3:
                return 0
            elif pos <= 10:
                return 1
            elif pos <= 15:
                return 2
            else:
                return 3
        
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        valid_mask = target_valid <= 21
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        pred_bins = np.array([position_to_bin(p) for p in pred_valid])
        target_bins = np.array([position_to_bin(t) for t in target_valid])
        
        correct = (pred_bins == target_bins).sum()
        total = len(target_valid)
        
        return correct / total if total > 0 else 0.0
    
    def compute_kendall_tau(self, predictions, targets, mask=None):
        # Compute Kendall's Tau correlation coefficient for ranking similarity
        from scipy.stats import kendalltau
        
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        # Only consider finishers (positions 1-20), exclude DNFs
        valid_mask = (target_valid >= 1) & (target_valid <= 20)
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        if len(pred_valid) < 2:
            return 0.0
        
        # Convert to rankings (lower position number = better rank)
        tau, _ = kendalltau(pred_valid, target_valid)
        return tau if not np.isnan(tau) else 0.0
    
    def compute_spearman_correlation(self, predictions, targets, mask=None):
        # Compute Spearman's rank correlation coefficient
        from scipy.stats import spearmanr
        
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        # Only consider finishers (positions 1-20), exclude DNFs
        valid_mask = (target_valid >= 1) & (target_valid <= 20)
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        if len(pred_valid) < 2:
            return 0.0
        
        corr, _ = spearmanr(pred_valid, target_valid)
        return corr if not np.isnan(corr) else 0.0
    
    def compute_ndcg(self, predictions, targets, mask=None, k=None):
        # Compute Normalized Discounted Cumulative Gain (NDCG)
        def dcg(scores, k=None):
            scores = np.asarray(scores)
            if k is not None:
                scores = scores[:k]
            gains = 2**scores - 1
            discounts = np.log2(np.arange(2, len(gains) + 2))
            return np.sum(gains / discounts)
        
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        # Only consider finishers (positions 1-20), exclude DNFs
        valid_mask = (target_valid >= 1) & (target_valid <= 20)
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        if len(pred_valid) < 2:
            return 0.0
        
        # Convert positions to relevance scores (higher position = lower score)
        # Position 1 gets score 20, position 20 gets score 1
        pred_scores = 21 - pred_valid
        target_scores = 21 - target_valid
        
        # Sort by predicted ranking
        pred_order = np.argsort(pred_valid)
        target_scores_sorted = target_scores[pred_order]
        
        dcg_score = dcg(target_scores_sorted, k)
        idcg_score = dcg(np.sort(target_scores)[::-1], k)  # Ideal DCG
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def compute_top_k_accuracy(self, predictions, targets, mask=None, k=3):
        # Compute accuracy of predicting top-k positions correctly
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        # Only consider finishers (positions 1-20), exclude DNFs
        valid_mask = (target_valid >= 1) & (target_valid <= 20)
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        if len(pred_valid) < k:
            return 0.0
        
        # Get top-k predicted positions
        pred_top_k = np.argsort(pred_valid)[:k] + 1  # +1 because positions start from 1
        target_top_k = np.argsort(target_valid)[:k] + 1
        
        # Compute intersection accuracy
        intersection = len(set(pred_top_k) & set(target_top_k))
        return intersection / k
    
    def compute_rank_biased_overlap(self, predictions, targets, mask=None, p=0.9):
        # Compute Rank Biased Overlap (RBO) similarity measure
        def rbo_score(list1, list2, p):
            if len(list1) != len(list2):
                raise ValueError("Lists must be same length")
            
            if not list1 or not list2:
                return 0.0
            
            k = len(list1)
            rbo = 0.0
            
            for d in range(1, k + 1):
                x_d = len(set(list1[:d]) & set(list2[:d]))
                rbo += (p ** (d - 1)) * (x_d / d)
            
            return (1 - p) * rbo
        
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        # Only consider finishers (positions 1-20), exclude DNFs
        valid_mask = (target_valid >= 1) & (target_valid <= 20)
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        if len(pred_valid) < 2:
            return 0.0
        
        # Convert to driver indices sorted by position (best to worst)
        pred_order = np.argsort(pred_valid)
        target_order = np.argsort(target_valid)
        
        return rbo_score(pred_order.tolist(), target_order.tolist(), p)
    
    def compute_mean_reciprocal_rank(self, predictions, targets, mask=None):
        # Compute Mean Reciprocal Rank (MRR)
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        # Only consider finishers (positions 1-20), exclude DNFs
        valid_mask = (target_valid >= 1) & (target_valid <= 20)
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        if len(pred_valid) < 1:
            return 0.0
        
        # For each actual position, find its rank in predictions
        mrr_sum = 0.0
        for actual_pos in target_valid:
            # Find where this position appears in predictions
            pred_ranks = np.argsort(pred_valid) + 1  # 1-based ranks
            actual_rank_in_pred = np.where(pred_valid == actual_pos)[0]
            if len(actual_rank_in_pred) > 0:
                rank = actual_rank_in_pred[0] + 1  # 1-based
                mrr_sum += 1.0 / rank
        
        return mrr_sum / len(target_valid) if len(target_valid) > 0 else 0.0
    
    def compute_ranking_similarity_metrics(self, predictions, targets, mask=None):
        # Compute comprehensive ranking similarity metrics
        metrics = {}
        
        # Kendall's Tau
        metrics['kendall_tau'] = self.compute_kendall_tau(predictions, targets, mask)
        
        # Spearman's Correlation
        metrics['spearman_corr'] = self.compute_spearman_correlation(predictions, targets, mask)
        
        # NDCG scores
        metrics['ndcg@5'] = self.compute_ndcg(predictions, targets, mask, k=5)
        metrics['ndcg@10'] = self.compute_ndcg(predictions, targets, mask, k=10)
        metrics['ndcg_all'] = self.compute_ndcg(predictions, targets, mask, k=None)
        
        # Top-k accuracies
        metrics['top_1_acc'] = self.compute_top_k_accuracy(predictions, targets, mask, k=1)
        metrics['top_3_acc'] = self.compute_top_k_accuracy(predictions, targets, mask, k=3)
        metrics['top_5_acc'] = self.compute_top_k_accuracy(predictions, targets, mask, k=5)
        
        # Rank Biased Overlap
        metrics['rbo'] = self.compute_rank_biased_overlap(predictions, targets, mask)
        
        # Mean Reciprocal Rank
        metrics['mrr'] = self.compute_mean_reciprocal_rank(predictions, targets, mask)
        
        return metrics

    def average_pairwise_accuracy(self, predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray = None) -> float:
       
        per_race_accs = []
        n_races = predictions.shape[0]

        for r in range(n_races):
            pred = predictions[r]
            true = targets[r]
            if masks is not None:
                mask = masks[r].astype(bool)
            else:
                mask = np.ones_like(true, dtype=bool)

            idxs = np.where(mask)[0]
            m = len(idxs)
            if m < 2:
                continue

            # compute pairwise comparisons
            total_pairs = 0
            correct_pairs = 0
            for i in range(m):
                for j in range(i + 1, m):
                    ii = idxs[i]
                    jj = idxs[j]
                    t_diff = int(true[ii]) - int(true[jj])
                    p_diff = int(pred[ii]) - int(pred[jj])

                    # sign comparison
                    s_t = 0 if t_diff == 0 else (1 if t_diff > 0 else -1)
                    s_p = 0 if p_diff == 0 else (1 if p_diff > 0 else -1)

                    if s_t == s_p:
                        correct_pairs += 1
                    total_pairs += 1

            if total_pairs > 0:
                per_race_accs.append(correct_pairs / total_pairs)

        return float(np.mean(per_race_accs)) if per_race_accs else 0.0

    def displacement_at_k(self, predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray = None, k: int = 3) -> float:
       
        per_race_disp = []
        n_races = predictions.shape[0]

        for r in range(n_races):
            pred = predictions[r]
            true = targets[r]
            if masks is not None:
                mask = masks[r].astype(bool)
            else:
                mask = np.ones_like(true, dtype=bool)

            # indices of drivers with true position <= k and valid
            idxs = np.where((true <= k) & mask)[0]
            if idxs.size == 0:
                continue

            disp = np.abs(pred[idxs].astype(float) - true[idxs].astype(float)).mean()
            per_race_disp.append(disp)

        return float(np.mean(per_race_disp)) if per_race_disp else 0.0

    def compute_podium_metrics(self, predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray = None) -> dict:
        
        #DNFs predicted for true podium drivers are penalized by mapping to PENALTY_POSITION.
       
        PENALTY_POSITION = 30

        total_podium = 0
        correct_within2 = 0
        podium_dnf_errors = 0
        penalized_errors = []

        n_races = predictions.shape[0]

        for r in range(n_races):
            pred = predictions[r]
            true = targets[r]
            if masks is not None:
                mask = masks[r].astype(bool)
            else:
                mask = np.ones_like(true, dtype=bool)

            idxs = np.where((true >= 1) & (true <= 3) & mask)[0]
            if idxs.size == 0:
                continue

            for i in idxs:
                t = int(true[i])
                p = int(pred[i])
                total_podium += 1

                if p == 21:
                    podium_dnf_errors += 1
                    penalized_errors.append(abs(PENALTY_POSITION - t))
                    # automatically incorrect for within-2
                    continue

                # non-DNF
                if abs(p - t) <= 2:
                    correct_within2 += 1

                penalized_errors.append(abs(p - t))

        if total_podium > 0:
            podium_within_2_acc = correct_within2 / total_podium
            podium_dnf_rate = podium_dnf_errors / total_podium
            podium_penalized_mae = float(np.mean(penalized_errors)) if penalized_errors else 0.0
        else:
            podium_within_2_acc = 0.0
            podium_dnf_rate = 0.0
            podium_penalized_mae = 0.0

        return {
            'podium_within_2_acc': float(podium_within_2_acc),
            'podium_dnf_rate': float(podium_dnf_rate),
            'podium_penalized_mae': float(podium_penalized_mae)
        }
    
    def train_single_fold(self, X_train_fold, y_train_fold, X_val_fold, y_val_fold, params):
        # Train model on a single fold and return validation accuracy
        train_mask = self.create_mask(y_train_fold)
        val_mask = self.create_mask(y_val_fold)
        
        X_train_t = torch.from_numpy(X_train_fold).float()
        y_train_t = torch.from_numpy(y_train_fold).long()
        train_mask_t = torch.from_numpy(train_mask).float()
        
        X_val_t = torch.from_numpy(X_val_fold).float()
        y_val_t = torch.from_numpy(y_val_fold).long()
        val_mask_t = torch.from_numpy(val_mask).float()
        
        train_ds = TensorDataset(X_train_t, y_train_t, train_mask_t)
        val_ds = TensorDataset(X_val_t, y_val_t, val_mask_t)
        
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
        
        input_dim = X_train_fold.shape[2]
        model = PositionPredictionNetwork(
            input_dim=input_dim,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            activation=params['activation'],
            use_pointer=self.use_pointer,
            use_conditional=self.use_conditional
        ).to(self.device)
        
        # Calculate class weights with higher weight for DNF class
        y_fold_flat = y_train_fold.flatten()
        valid_mask = (y_fold_flat >= 1) & (y_fold_flat <= 21)
        y_fold_valid = y_fold_flat[valid_mask]
        class_counts = np.bincount((y_fold_valid - 1).astype(int), minlength=21)  # positions 1-21 -> classes 0-20
        total_samples = class_counts.sum()
        class_weights = total_samples / (class_counts + 1e-6)  # Inverse frequency weighting
        # Give DNF class (index 20) much higher weight
        class_weights[20] *= 10.0  # 5x weight for DNF class
        class_weights = class_weights / class_weights.sum() * 21  # Normalize
        class_weights = torch.from_numpy(class_weights).float().to(self.device)
        
        # Use Focal Loss for better handling of imbalanced classes
        class FocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0, reduction='none'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:  # 'none'
                    return focal_loss
        
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='none')
        
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                momentum=0.9
            )
        
        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            for xb, yb, mask_b in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mask_b = mask_b.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_conditional:
                    # Conditional modeling: dual outputs
                    position_logits, dnf_probs = model(xb)
                    
                    # Position loss
                    position_logits_flat = position_logits.view(-1, 21)
                    targets_flat = (yb - 1).clamp(0, 20).view(-1).long()
                    mask_flat = mask_b.view(-1)
                    position_loss = criterion(position_logits_flat, targets_flat)
                    
                    # DNF loss (BCE)
                    dnf_targets = (yb == 21).float().view(-1, 1)  # 1 if DNF, 0 otherwise
                    dnf_loss = nn.functional.binary_cross_entropy(
                        dnf_probs.view(-1, 1), dnf_targets, reduction='none'
                    ).view(-1)
                    
                    # Combine losses with masking
                    total_loss_per_sample = position_loss + dnf_loss.view(-1)
                    loss = (total_loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                else:
                    # Standard modeling
                    logits = model(xb)
                    logits_flat = logits.view(-1, 21)
                    targets_flat = (yb - 1).clamp(0, 20).view(-1).long()
                    mask_flat = mask_b.view(-1)
                    
                    loss_per_sample = criterion(logits_flat, targets_flat)
                    loss = (loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation fold
        model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for xb, yb, mask_b in val_loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=-1)
                
                assignments = self.hungarian_matching(probs, mask_b)
                assignments = self.adjust_positions_for_dnfs(assignments)
                
                all_preds.append(assignments)
                all_targets.append(yb.numpy())
                all_masks.append(mask_b.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        tol2_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=2)
        return tol2_acc
    
    def cross_validate(self, params, n_folds=5):
        # Perform k-fold cross-validation and return average accuracy
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        fold_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            fold_acc = self.train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, params
            )
            fold_accs.append(fold_acc)
        
        return np.mean(fold_accs)
    
    def grid_search(self, param_grid):
        # Perform grid search with 5-fold cross-validation
        print("="*80)
        print("Starting Grid Search with 5-Fold Cross-Validation")
        print("="*80)
        
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        total_combinations = len(param_combinations)
        print(f"\nTotal parameter combinations to test: {total_combinations}\n")
        
        best_score = -1
        best_params = None
        
        for i, params in enumerate(param_combinations, 1):
            print(f"[{i}/{total_combinations}] Testing: "
                  f"layers={params['hidden_layers']}, "
                  f"dropout={params['dropout']}, "
                  f"act={params['activation']}, "
                  f"lr={params['lr']}, "
                  f"opt={params['optimizer']}, "
                  f"bs={params['batch_size']}, "
                  f"wd={params['weight_decay']}, "
                  f"epochs={params['epochs']}")
            
            try:
                avg_tol2_acc = self.cross_validate(params)
                print(f"  → CV Accuracy (±2 positions): {avg_tol2_acc*100:.2f}%")
                
                if avg_tol2_acc > best_score:
                    best_score = avg_tol2_acc
                    best_params = params.copy()
                    print(f"  New best!")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print("\n" + "="*80)
        print("Grid Search Complete!")
        print("="*80)
        print(f"\nBest parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest CV Accuracy (±2 positions): {best_score*100:.2f}%")
        print("="*80 + "\n")
        
        self.best_params = best_params
        self.best_cv_score = best_score
        
        return best_params, best_score
    
    def train_final_model(self, params):
        # Train final model on all training data
        print("\nTraining final model on all training data...")
        print(f"  Hidden layers: {params['hidden_layers']}")
        print(f"  Dropout: {params['dropout']}")
        print(f"  Activation: {params['activation']}")
        print(f"  Learning rate: {params['lr']}")
        print(f"  Optimizer: {params['optimizer']}")
        print(f"  Batch size: {params['batch_size']}")
        print(f"  Weight decay: {params['weight_decay']}")
        print(f"  Epochs: {params['epochs']}")
        print(f"  Device: {self.device}\n")
        
        train_mask = self.create_mask(self.y_train)
        
        X_train_t = torch.from_numpy(self.X_train).float()
        y_train_t = torch.from_numpy(self.y_train).long()
        train_mask_t = torch.from_numpy(train_mask).float()
        
        train_ds = TensorDataset(X_train_t, y_train_t, train_mask_t)
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        
        input_dim = self.X_train.shape[2]
        self.model = PositionPredictionNetwork(
            input_dim=input_dim,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            activation=params['activation'],
            use_pointer=self.use_pointer,
            use_conditional=self.use_conditional
        ).to(self.device)
        
        # Calculate class weights with higher weight for DNF class
        y_flat = self.y_train.flatten()
        valid_mask = (y_flat >= 1) & (y_flat <= 21)
        y_valid = y_flat[valid_mask]
        class_counts = np.bincount((y_valid - 1).astype(int), minlength=21)  # positions 1-21 -> classes 0-20
        total_samples = class_counts.sum()
        class_weights = total_samples / (class_counts + 1e-6)  # Inverse frequency weighting
        # Give DNF class (index 20) much higher weight
        class_weights[20] *= 10.0  # 5x weight for DNF class
        class_weights = class_weights / class_weights.sum() * 21  # Normalize
        class_weights = torch.from_numpy(class_weights).float().to(self.device)
        
        
        # Use Focal Loss for better handling of imbalanced classes
        class FocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0, reduction='none'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:  # 'none'
                    return focal_loss
        
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='none')
        
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                momentum=0.9
            )
        # reset theta history for this training run
        self.theta_history = []
        
        # Training loop
        for epoch in range(1, params['epochs'] + 1):
            self.model.train()
            train_losses = []
            
            for xb, yb, mask_b in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mask_b = mask_b.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_conditional:
                    # Conditional modeling: dual outputs
                    position_logits, dnf_probs = self.model(xb)
                    
                    # Position loss
                    position_logits_flat = position_logits.view(-1, 21)
                    targets_flat = (yb - 1).clamp(0, 20).view(-1).long()
                    mask_flat = mask_b.view(-1)
                    position_loss = criterion(position_logits_flat, targets_flat)
                    
                    # DNF loss (BCE)
                    dnf_targets = (yb == 21).float().view(-1, 1)  # 1 if DNF, 0 otherwise
                    dnf_loss = nn.functional.binary_cross_entropy(
                        dnf_probs.view(-1, 1), dnf_targets, reduction='none'
                    ).view(-1)
                    
                    # Combine losses with masking
                    total_loss_per_sample = position_loss + dnf_loss.view(-1)
                    loss = (total_loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                else:
                    # Standard modeling
                    logits = self.model(xb)
                    logits_flat = logits.view(-1, 21)
                    targets_flat = (yb - 1).clamp(0, 20).view(-1).long()
                    mask_flat = mask_b.view(-1)
                    
                    loss_per_sample = criterion(logits_flat, targets_flat)
                    loss = (loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            # record parameter vector for this epoch (for f(theta) visualization)
            try:
                theta_vec = self._flatten_params(self.model)
                self.theta_history.append(theta_vec)
            except Exception:
                # if something goes wrong recording params, continue without failing training
                pass
            
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f}")
        
        return self.model
    
    def test_model(self, show_predictions=True):
        # Evaluate model on test set
        print("\n" + "="*80)
        print("Final Test Performance")
        print("="*80)
        
        test_mask = self.create_mask(self.y_test)
        
        X_test_t = torch.from_numpy(self.X_test).float()
        y_test_t = torch.from_numpy(self.y_test).long()
        test_mask_t = torch.from_numpy(test_mask).float()
        
        test_ds = TensorDataset(X_test_t, y_test_t, test_mask_t)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for xb, yb, mask_b in test_loader:
                xb = xb.to(self.device)
                
                if self.use_pointer:
                    # Pointer network outputs assignments directly
                    pointers = self.model(xb)  # [batch, num_drivers] - driver indices in order
                    assignments = self.pointer_to_assignments(pointers.cpu().numpy())
                elif self.use_conditional:
                    # Conditional modeling: use position predictions (DNF conditioning is handled in training)
                    position_logits, dnf_probs = self.model(xb)
                    position_probs = torch.softmax(position_logits, dim=-1)  # [batch, drivers, 21]
                    assignments = self.hungarian_matching(position_probs, mask_b)
                else:
                    # Standard modeling
                    logits = self.model(xb)
                    probs = torch.softmax(logits, dim=-1)
                    assignments = self.hungarian_matching(probs, mask_b)
                
                assignments = self.adjust_positions_for_dnfs(assignments)
                
                all_preds.append(assignments)
                all_targets.append(yb.numpy())
                all_masks.append(mask_b.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        # Print simplified race summary
        if show_predictions:
            total_races = len(self.race_metadata)
            test_start_idx = total_races - len(all_preds)
            
            # Count races with DNFs
            races_with_dnfs = 0
            total_predicted_dnfs = 0
            
            for i in range(len(all_preds)):
                race_preds = all_preds[i]
                dnf_count = (race_preds == 21).sum()
                if dnf_count > 0:
                    races_with_dnfs += 1
                    total_predicted_dnfs += dnf_count
            
            print(f"\nTest Set Summary:")
            print(f"- {len(all_preds)} races evaluated")
            print(f"- {races_with_dnfs} races predicted to have DNFs ({total_predicted_dnfs} total DNF predictions)")
            print(f"- Average DNFs per race: {total_predicted_dnfs/len(all_preds):.1f}")
            print(f"- Actual DNF rate in test data: {(all_targets == 21).sum() / all_targets.size * 100:.1f}%")
        
        # Overall metrics
        exact_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=0)
        tol1_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=1)
        tol2_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=2)
        tol3_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=3)
        bin_acc = self.compute_bin_accuracy(all_preds, all_targets, all_masks)
        
        valid_mask = (all_targets <= 20) & (all_masks == 1)
        mae = mean_absolute_error(
            all_targets[valid_mask].flatten(),
            all_preds[valid_mask].flatten()
        )
        
        print(f"\n{'='*80}")
        print(f"Overall Test Performance ({len(all_preds)} races):")
        print(f"{'='*80}")
        print(f"Test Exact Position Accuracy: {exact_acc * 100:.2f}%")
        print(f"Test Accuracy (±1 position): {tol1_acc * 100:.2f}%")
        print(f"Test Accuracy (±2 positions): {tol2_acc * 100:.2f}%")
        print(f"Test Accuracy (±3 positions): {tol3_acc * 100:.2f}%")
        print(f"Test Bin Accuracy (Podium/Points/Mid/Back): {bin_acc * 100:.2f}%")
        print(f"Test Mean Absolute Error: {mae:.2f} positions")
        
        # Ranking Similarity Metrics
        print(f"\n{'='*80}")
        print("RANKING SIMILARITY METRICS")
        print(f"{'='*80}")
        
        ranking_metrics = self.compute_ranking_similarity_metrics(all_preds, all_targets, all_masks)
        
        print(f"Kendall's Tau Correlation: {ranking_metrics['kendall_tau']:.4f}")
        print(f"Spearman's Rank Correlation: {ranking_metrics['spearman_corr']:.4f}")
        print(f"NDCG@5: {ranking_metrics['ndcg@5']:.4f}")
        print(f"NDCG@10: {ranking_metrics['ndcg@10']:.4f}")
        print(f"NDCG@All: {ranking_metrics['ndcg_all']:.4f}")
        print(f"Top-1 Accuracy: {ranking_metrics['top_1_acc']:.4f}")
        print(f"Top-3 Accuracy: {ranking_metrics['top_3_acc']:.4f}")
        print(f"Top-5 Accuracy: {ranking_metrics['top_5_acc']:.4f}")
        print(f"Rank Biased Overlap (RBO): {ranking_metrics['rbo']:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {ranking_metrics['mrr']:.4f}")
        
        # PAIRWISE AND TOP-K RANK ACCURACY METRICS
        print(f"\n{'='*80}")
        print("PAIRWISE AND TOP-K RANK ACCURACY METRICS")
        print(f"{'='*80}")

        try:
            pairwise_acc = self.average_pairwise_accuracy(all_preds, all_targets, all_masks)
            disp_at_3 = self.displacement_at_k(all_preds, all_targets, all_masks, k=3)
            disp_at_5 = self.displacement_at_k(all_preds, all_targets, all_masks, k=5)

            print(f"Average Pairwise Accuracy (mean over races): {pairwise_acc * 100:.2f}%")
            print(f"Displacement@3 (mean abs error for true top-3): {disp_at_3:.2f} positions")
            print(f"Displacement@5 (mean abs error for true top-5): {disp_at_5:.2f} positions")
        except Exception as e:
            print(f"Failed to compute pairwise/top-k metrics: {e}")

        # PODIUM-FOCUSED METRICS (WITH DNF PENALTY)
        print(f"\n{'='*80}")
        print("PODIUM-FOCUSED METRICS (WITH DNF PENALTY)")
        print(f"{'='*80}")
        try:
            podium_metrics = self.compute_podium_metrics(all_preds, all_targets, all_masks)
            print(f"Podium within ±2 positions (no DNFs allowed): {podium_metrics['podium_within_2_acc'] * 100:.2f}%")
            print(f"Podium DNF rate (true podium predicted as DNF): {podium_metrics['podium_dnf_rate'] * 100:.2f}%")
            print(f"Podium penalized MAE (DNF mapped to position 30): {podium_metrics['podium_penalized_mae']:.2f} positions")
        except Exception as e:
            print(f"Failed to compute podium metrics: {e}")

        # Production-grade evaluation metrics
        print(f"\n{'='*80}")
        print("PRODUCTION-GRADE EVALUATION METRICS")
        print(f"{'='*80}")

        self._production_evaluation_metrics(all_preds, all_targets, all_masks, tol2_acc)
        
        print("="*80 + "\n")
        
        self.test_tol2_acc = tol2_acc
        self.test_mae = mae
        
        return tol2_acc, mae

    def visualize_loss_landscape_f_theta(self, sample_size: int = 50, grid_size: int = 21,
                                         alpha: float = 1.0,
                                         save_path: str = "loss_landscape_f_theta.png") -> None:
        # Visualize f(theta): loss as a function of parameters in a 2D PCA plane.
        if not self.theta_history:
            print("No theta history found. Train the model first.")
            return

        print("\nGenerating f(theta) loss landscape visualization...")

        # stack parameter vectors: shape (epochs, num_params)
        theta_mat = np.stack(self.theta_history, axis=0)

        # PCA in parameter space
        pca = PCA(n_components=2)
        pca.fit(theta_mat)

        theta_star = theta_mat[-1]     # final parameters
        d1, d2 = pca.components_       # two principal directions

        #  small subset of data to estimate loss
        X = self.X_train
        y = self.y_train
        mask = self.create_mask(y)

        if X.shape[0] > sample_size:
            idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
            X = X[idx]
            y = y[idx]
            mask = mask[idx]

        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        mask_t = torch.from_numpy(mask).float().to(self.device)

        
        def load_flat_params_into_model(flat_theta: np.ndarray) -> None:
            flat = torch.from_numpy(flat_theta).to(self.device)
            offset = 0
            with torch.no_grad():
                for p in self.model.parameters():
                    numel = p.numel()
                    new_vals = flat[offset:offset + numel].view_as(p)
                    p.copy_(new_vals)
                    offset += numel

        #evaluate loss on a 2D grid in PCA coordinates
        us = np.linspace(-alpha, alpha, grid_size)
        vs = np.linspace(-alpha, alpha, grid_size)
        U, V = np.meshgrid(us, vs)
        Z = np.zeros_like(U)

        self.model.eval()
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    u = U[i, j]
                    v = V[i, j]

                    theta_uv = theta_star + u * d1 + v * d2
                    load_flat_params_into_model(theta_uv)

                    logits = self.model(X_t)  # [batch, drivers, 21]
                    logits_flat = logits.view(-1, 21)
                    targets_flat = (y_t - 1).clamp(0, 20).view(-1)
                    mask_flat = mask_t.view(-1)

                    ce = nn.functional.cross_entropy(
                        logits_flat, targets_flat, reduction='none'
                    )
                    loss = (ce * mask_flat).sum().item() / (mask_flat.sum().item() + 1e-8)
                    Z[i, j] = loss

        # project training trajectory into PCA plane
        traj_2d = pca.transform(theta_mat)  # [epochs, 2]

        # plot surface + trajectory
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U, V, Z, linewidth=0, antialiased=True, alpha=0.75)
        ax.plot(
            traj_2d[:, 0],
            traj_2d[:, 1],
            [Z.min() - 0.1] * len(traj_2d),
            marker='o',
            color='black',
            label='Training trajectory',
        )

        ax.set_xlabel("PCA dir 1 in parameter space")
        ax.set_ylabel("PCA dir 2 in parameter space")
        ax.set_zlabel("Loss")
        ax.set_title("Loss landscape f(θ) with training trajectory")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved f(theta) loss landscape figure to {save_path}")

    def visualize_prediction_surface_g_x(self, grid_size: int = 51, alpha: float = 2.0,
                                         save_path: str = "prediction_surface_g_x.png") -> None:
        # Visualize g(x): model predictions as a function of input features in a 2D PCA plane.
        print("\nGenerating g(x) prediction surface visualization...")

        if self.use_pointer or self.use_conditional:
            print("g(x) visualization currently implemented only for standard (non-pointer, non-conditional) model.")
            return

        #  flatten all driver features from training set
        num_races, num_drivers, num_features = self.X_train.shape
        X_flat = self.X_train.reshape(-1, num_features)  # [num_races * drivers, num_features]

        # PCA in feature space
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_flat)
        mean_2d = X_2d.mean(axis=0)
        std_2d = X_2d.std(axis=0)

        #  build grid in PCA space around mean
        u_vals = np.linspace(mean_2d[0] - alpha * std_2d[0],
                             mean_2d[0] + alpha * std_2d[0], grid_size)
        v_vals = np.linspace(mean_2d[1] - alpha * std_2d[1],
                             mean_2d[1] + alpha * std_2d[1], grid_size)
        U, V = np.meshgrid(u_vals, v_vals)
        grid_points_2d = np.stack([U.ravel(), V.ravel()], axis=1)  # [G, 2]

        # 3) map grid back to original feature space
        grid_features = pca.inverse_transform(grid_points_2d)  # [G, num_features]
        grid_features_t = torch.from_numpy(grid_features).float().to(self.device)

        # 4) run through encoder + position head
        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encoder(grid_features_t)   # [G, hidden_dim]
            logits = self.model.position_head(encoded)      # [G, 21]
            probs = torch.softmax(logits, dim=-1)           # [G, 21]
            
            # expected finishing position: sum_k p_k * k
            positions = torch.arange(1, 22, device=self.device).float()
            expected_position = (probs * positions).sum(dim=-1)  # [G]
            expected_position_np = expected_position.cpu().numpy()

        Z = expected_position_np.reshape(U.shape)

        # plot surface
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U, V, Z, linewidth=0, antialiased=True)
        ax.set_xlabel("PC1 of driver feature space")
        ax.set_ylabel("PC2 of driver feature space")
        ax.set_zlabel("Expected finishing position")
        ax.set_title("Prediction surface g(x) in PCA driver-feature space")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Saved g(x) prediction surface figure to {save_path}")
    
    def _production_evaluation_metrics(self, predictions, targets, masks, tol2_acc):
        # Comprehensive production-grade evaluation metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        import numpy as np
        
        # Flatten predictions and targets
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        mask_flat = masks.flatten()
        
        # Only evaluate valid predictions
        valid = mask_flat == 1
        pred_valid = pred_flat[valid]
        target_valid = target_flat[valid]
        
        # 1. DNF-Specific Metrics
        print("1. DNF PREDICTION METRICS")
        print("-" * 40)
        
        # Binary classification: DNF (21) vs Finish (1-20)
        pred_dnf = (pred_valid == 21).astype(int)
        actual_dnf = (target_valid == 21).astype(int)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            actual_dnf, pred_dnf, average='binary', zero_division=0
        )
        
        print(f"DNF Precision: {precision:.3f} (Predicted DNFs that were actually DNFs)")
        print(f"DNF Recall: {recall:.3f} (Actual DNFs that were predicted as DNFs)")
        print(f"DNF F1-Score: {f1:.3f} (Harmonic mean of precision/recall)")
        print(f"Actual DNFs: {actual_dnf.sum()}, Predicted DNFs: {pred_dnf.sum()}")
        
        # 2. Position Range Analysis
        print(f"\n2. POSITION RANGE ANALYSIS")
        print("-" * 40)
        
        def get_position_range(pos):
            if pos <= 3:
                return "Podium"
            elif pos <= 10:
                return "Points"
            elif pos <= 15:
                return "Midfield"
            elif pos == 21:
                return "DNF"
            else:
                return "Back"
        
        pred_ranges = [get_position_range(p) for p in pred_valid]
        actual_ranges = [get_position_range(t) for t in target_valid]
        
        range_precision, range_recall, range_f1, _ = precision_recall_fscore_support(
            actual_ranges, pred_ranges, average=None, 
            labels=["Podium", "Points", "Midfield", "Back", "DNF"]
        )
        
        ranges = ["Podium", "Points", "Midfield", "Back", "DNF"]
        for i, r in enumerate(ranges):
            print(f"{r:8}: Precision={range_precision[i]:.3f}, Recall={range_recall[i]:.3f}, F1={range_f1[i]:.3f}")
        
        # 3. Confusion Matrix Analysis (Top 10 positions + DNF)
        print(f"\n3. CONFUSION MATRIX SUMMARY (Positions 1-10 + DNF)")
        print("-" * 50)
        
        # Focus on positions 1-10 and DNF
        focus_mask = ((target_valid >= 1) & (target_valid <= 10)) | (target_valid == 21)
        focus_pred = pred_valid[focus_mask]
        focus_target = target_valid[focus_mask]
        
        # Map DNF to position 11 for confusion matrix
        focus_pred_cm = focus_pred.copy()
        focus_target_cm = focus_target.copy()
        focus_pred_cm[focus_pred_cm == 21] = 11
        focus_target_cm[focus_target_cm == 21] = 11
        
        cm = confusion_matrix(focus_target_cm, focus_pred_cm, labels=list(range(1, 12)))
        
        # Show diagonal accuracy for key positions
        diagonal = np.diag(cm)
        total_per_class = cm.sum(axis=1)
        
        print("Position | Accuracy | Total Predictions")
        print("-" * 35)
        for i in range(10):  # Positions 1-10
            acc = diagonal[i] / total_per_class[i] if total_per_class[i] > 0 else 0
            
        dnf_acc = diagonal[10] / total_per_class[10] if total_per_class[10] > 0 else 0
        
        
        # 4. Feature Usage Analysis
        print(f"\n4. FEATURE USAGE VALIDATION")
        print("-" * 40)
        
        # Check if DNF features are in the feature set
        dnf_features = [i for i, name in enumerate(self.feature_names) if 'dnf' in name.lower()]
        if dnf_features:
            print(f"DNF features found: {[self.feature_names[i] for i in dnf_features]}")
            print("Model has access to DNF historical data")

            # Analyze feature correlations with DNF outcomes
            X_test_flat = self.X_test.reshape(-1, self.X_test.shape[2])
            y_test_flat = self.y_test.flatten()
            mask_flat = self.create_mask(self.y_test).flatten()

            valid = mask_flat == 1
            dnf_outcomes = (y_test_flat == 21)[valid]

            print("\nDNF Feature Correlations with Actual DNFs:")
            for idx in dnf_features:
                feature_values = X_test_flat[valid, idx]
                correlation = np.corrcoef(feature_values, dnf_outcomes.astype(int))[0, 1]
                
        else:
            print("No DNF features found in feature set")
        
        # 5. Model Calibration Check
        print(f"\n5. MODEL CALIBRATION ANALYSIS")
        print("-" * 40)
        
        # For positions 1-5, check if predictions are reasonable
        podium_mask = (target_valid >= 1) & (target_valid <= 3)
        if podium_mask.sum() > 0:
            podium_preds = pred_valid[podium_mask]
            podium_targets = target_valid[podium_mask]
            
            # Check how often podium predictions are reasonable
            reasonable_podium = ((podium_preds >= 1) & (podium_preds <= 5)).sum()
            print(f"Podium predictions in top-5 range: {reasonable_podium}/{podium_mask.sum()} ({reasonable_podium/podium_mask.sum()*100:.1f}%)")
        
        # 6. Race-Level Consistency
        print(f"\n6. RACE-LEVEL CONSISTENCY")
        print("-" * 40)
        
        race_maes = []
        race_dnf_accuracy = []
        
        for race_idx in range(len(predictions)):
            race_pred = predictions[race_idx]
            race_target = targets[race_idx]
            race_mask = masks[race_idx]
            
            # Race MAE (excluding DNFs)
            valid_positions = (race_target <= 20) & (race_mask == 1)
            if valid_positions.sum() > 0:
                race_mae = np.mean(np.abs(race_pred[valid_positions] - race_target[valid_positions]))
                race_maes.append(race_mae)
            
            # DNF accuracy for this race
            race_dnf_pred = (race_pred == 21)
            race_dnf_actual = (race_target == 21)
            race_dnf_acc = np.mean(race_dnf_pred == race_dnf_actual)
            race_dnf_accuracy.append(race_dnf_acc)
        
        if race_maes:
            print(f"Average Race MAE: {np.mean(race_maes):.2f} ± {np.std(race_maes):.2f}")
        print(f"Average Race DNF Accuracy: {np.mean(race_dnf_accuracy)*100:.1f}% ± {np.std(race_dnf_accuracy)*100:.1f}%")
        
        # 7. Production Readiness Score
        print(f"\n7. PRODUCTION READINESS SCORE")
        print("-" * 40)
        
        # Calculate a composite score
        base_score = tol2_acc * 100  # ±2 position accuracy as base
        
        # Bonuses
        dnf_bonus = min(f1 * 20, 10)  # Up to 10 points for good DNF prediction
        consistency_bonus = max(0, 5 - np.std(race_maes)) if race_maes else 0  # Up to 5 points for consistency
        
        total_score = base_score + dnf_bonus + consistency_bonus
        
        print(f"Base Score (±2 accuracy): {base_score:.1f}/100")
        print(f"DNF Bonus: +{dnf_bonus:.1f}/10")
        print(f"Consistency Bonus: +{consistency_bonus:.1f}/5")
        print(f"Total Production Score: {total_score:.1f}/115")
        
        if total_score >= 80:
            print("PRODUCTION READY: Excellent performance across all metrics")
        elif total_score >= 65:
            print("GOOD: Solid performance, ready for production with monitoring")
        elif total_score >= 50:
            print("NEEDS IMPROVEMENT: Functional but requires optimization")
        else:
            print("NOT READY: Significant issues need addressing")


def main():
    DATA_FILE = "f1_processed_2021_2025.npz"
    RUN_GRID_SEARCH = False
    
    # Initialize pipeline
    pipeline = F1PositionPredictionPipeline(
        data_file=DATA_FILE,
        device='cuda',
        seed=339,
        use_pointer=False,  # Set to True to use Pointer Network instead of Hungarian
        use_conditional=False  # Set to True to use conditional DNF modeling
    )
    
    # Load data
    pipeline.load_data()
    
    # Define hyperparameter grid
    param_grid = {
        'hidden_layers': [(128, 64), (64, 32)],
        'dropout': [0.2],
        'activation': ['relu', 'leaky_relu'],
        'lr': [1e-3, 5e-4],
        'optimizer': ['adam'],
        'batch_size': [16, 32],
        'weight_decay': [1e-4],
        'epochs': [50]
    }
    
    if RUN_GRID_SEARCH:
        # Grid search with cross-validation
        best_params, best_cv_score = pipeline.grid_search(param_grid)
    else:
        # Use preset best parameters
        best_params = {
            'hidden_layers': (64, 32),
            'dropout': 0.2,
            'activation': 'relu',
            'lr': 1e-3,
            'optimizer': 'adam',
            'batch_size': 32,
            'weight_decay': 1e-4,
            'epochs': 50
        }
        print("\nUsing best hyperparameters from previous grid search")
    
    # Train final model
    pipeline.train_final_model(best_params)
    # Visualizations for requirement 4
    pipeline.visualize_loss_landscape_f_theta()
    pipeline.visualize_prediction_surface_g_x()
    
    # Test on held-out test set
    pipeline.test_model(show_predictions=True)


if __name__ == "__main__":
    main()
