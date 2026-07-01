import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
import numpy as np
import time
from collections import deque
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Sample import FeatureImportancePCA, SampleHierarchy
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from idec import IDEC
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, Birch

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ClusteringState:
    def __init__(self, history_length=3):
        self.history_length = history_length
        self.reset()
    def reset(self):
        self.nmi_history = deque(maxlen=self.history_length)
        self.ari_history = deque(maxlen=self.history_length)
        self.jaccard_history = deque(maxlen=self.history_length)
        self.center_move_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        self.consistency_history = deque(maxlen=self.history_length)
        self.sample_ratio = 0.0
        self.feature_ratio = 0.0
        self.iteration = 0
        self.last_action_type = -1
        self.last_action_ratio = 0.0
        self.last_reward = 0.0
        self.mean_time = 0.0
        self.std_time = 0.01
    def get_state_vector(self):
        state = []
        consistency = self.consistency_history[-1] if self.consistency_history else 0
        state.append(consistency)
        nmi = self.nmi_history[-1] if self.nmi_history else 0
        state.append(nmi)
        jaccard = self.jaccard_history[-1] if self.jaccard_history else 0
        state.append(jaccard)
        center_move = self.center_move_history[-1] if self.center_move_history else 0
        state.append(center_move)
        current_time = self.time_history[-1] if self.time_history else 0
        if self.std_time > 0:
            z_time = (current_time - self.mean_time) / self.std_time
        else:
            z_time = 0
        state.append(np.tanh(z_time / 2))
        state.append(self.sample_ratio)
        state.append(self.feature_ratio)
        state.append(self.iteration / 100)
        action_type_norm = (self.last_action_type * 2) - 1 if self.last_action_type >= 0 else 0
        state.append(action_type_norm)
        state.append(self.last_action_ratio)
        state.append(self.last_reward)
        return np.array(state, dtype=np.float32)

class RewardFunction:
    def __init__(self):
        self.quality_history = []
        self.nmi_history = []
        self.jaccard_history = []
        self.center_move_history = []
        self.time_history = []
    def compute_reward(self, state, next_state, action, time_cost):
        reward = 0
        if len(state.nmi_history) > 0 and len(next_state.nmi_history) > 0:
            current_nmi = next_state.nmi_history[-1]
            prev_nmi = state.nmi_history[-1]
            delta_nmi = current_nmi - prev_nmi
            if action == 0:
                nmi_reward = delta_nmi * 30
            else:
                nmi_reward = delta_nmi * 30
            reward += nmi_reward
        reward -= 0.2
        curr_const = next_state.consistency_history[-1] if len(next_state.consistency_history) > 0 else 0.0
        prev_const = state.consistency_history[-1] if len(state.consistency_history) > 0 else 0.0
        consistency_reward = curr_const - prev_const
        reward += consistency_reward
        current_jaccard = next_state.jaccard_history[-1] if next_state.jaccard_history else 0
        jaccard_reward = 0.0
        if current_jaccard > 0.95:
            jaccard_reward = 0.1
        reward += jaccard_reward
        center_move = next_state.center_move_history[-1] if next_state.center_move_history else 1.0
        if current_nmi > 0.85 and center_move < 0.001:
            reward += 10.0
        if len(state.time_history) > 0:
            time_mean = np.mean(list(state.time_history))
            time_ratio = time_cost / max(time_mean, 0.01)
            if time_ratio > 1.5:
                time_penalty = min((time_ratio - 1.5) * 0.2, 0.5)
                reward -= time_penalty
        return reward

class ExperienceBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(dones))
    def __len__(self):
        return len(self.buffer)

class PPOAgent:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10):
        self.policy_net = PolicyNetwork(state_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.buffer = ExperienceBuffer()
    def select_action(self, state, epsilon=0.001):
        if np.random.random() < epsilon:
            action = np.random.choice(2)
            return action
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            action_probs = action_probs.squeeze().cpu().numpy()
        if np.any(np.isnan(action_probs)) or np.any(np.isinf(action_probs)):
            action_probs = np.array([0.5, 0.5])
        else:
            action_probs = np.clip(action_probs, 1e-10, 1.0)
            prob_sum = np.sum(action_probs)
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                action_probs = np.array([0.5, 0.5])
        try:
            action = np.random.choice(2, p=action_probs)
        except ValueError:
            action = np.random.choice(2)
        return action
    def update(self, batch_size=32):
        if len(self.buffer) < 1:
            return
        batch_size = min(batch_size, len(self.buffer))
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        with torch.no_grad():
            old_action_probs = self.policy_net(states)
            old_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
            next_values = self.value_net(next_states).squeeze(1)
            targets = rewards + self.gamma * next_values * (1 - dones)
        for _ in range(self.epochs):
            action_probs = self.policy_net(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
            values = self.value_net(states).squeeze(1)
            ratio = torch.exp(log_probs - old_log_probs)
            advantages = (targets - values).detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, targets)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
        print(f"  policy_loss: {policy_loss.item():.4f}, value_loss: {value_loss.item():.4f}")

def evaluate_clustering(X, labels):
    if len(np.unique(labels)) < 2:
        return 0
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = 0
    return silhouette

def compute_jaccard_similarity(labels1, labels2):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels1, labels2)

def compute_center_movement(centers1, centers2):
    if centers1 is None or centers2 is None:
        return 0
    min_len = min(len(centers1), len(centers2))
    if min_len == 0:
        return 0
    distances = []
    for i in range(min_len):
        dist = np.linalg.norm(centers1[i] - centers2[i])
        distances.append(dist)
    return np.mean(distances)

def compute_pairwise_jaccard(labels1, labels2):
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    c_matrix = contingency_matrix(labels1, labels2)
    intersection = np.sum(c_matrix * (c_matrix - 1))

    a_i = np.sum(c_matrix, axis=1)
    pairs_in_labels1 = np.sum(a_i * (a_i - 1))
    b_j = np.sum(c_matrix, axis=0)
    pairs_in_labels2 = np.sum(b_j * (b_j - 1))
    union = pairs_in_labels1 + pairs_in_labels2 - intersection
    return intersection / union if union > 0 else 0.0

class RLClustering:
    def __init__(self, X, y_true=None, feature_importance=None, sample_hierarchy=None, method='kmeans'):
        self.X = X
        self.y_true = y_true
        self.feature_importance = feature_importance
        self.sample_hierarchy = sample_hierarchy
        self.method = method
        self.best_idec = None
        self.best_cdc = None
        self.best_features = None
        self.cdc_model = None
        self.cdc_initialized = False
        if hasattr(self, 'idec'):
            del self.idec
            self.idec = None
        self.agent = PPOAgent(state_dim=11, lr=3e-4, gamma=0.99, epochs=10)
        self.reward_fn = RewardFunction()
        self.state = ClusteringState()
        self.global_best_nmi = -1
        self.global_best_samples = None
        self.global_best_features = None
        self.global_best_labels = None
        self.global_best_centers = None
        self.selected_samples = None
        self.selected_features = None
        self.current_labels = None
        self.current_centers = None
        self.zeus_model = None
        self.history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'nmi': [],
            'sample_ratio': [],
            'feature_ratio': [],
            'time': []
        }

    def reset_env(self):
        if self.sample_hierarchy is not None:
            self.sample_hierarchy.selected_mask = np.zeros(self.X.shape[0], dtype=bool)
        self.state.reset()
        self.selected_samples = None
        self.selected_features = None
        self.current_labels = None
        self.current_centers = None
        self.initialize_selection()
        k = len(np.unique(self.y_true))
        if len(self.selected_features) > 0 and len(self.selected_samples) > 0:
            km = KMeans(n_clusters=k, n_init=1, random_state=None)
            X_selected = self.X[self.selected_samples][:, self.selected_features]
            self.current_labels = km.fit_predict(X_selected)
            self.current_centers = km.cluster_centers_
            from sklearn.metrics import normalized_mutual_info_score
            y_curr = self.y_true[self.selected_samples]
            init_nmi = normalized_mutual_info_score(y_curr, self.current_labels)
            print(f"使用前10%特征的KMeans NMI: {init_nmi:.4f}")
            self.state.nmi_history.append(init_nmi)
            self.state.jaccard_history.append(1.0)
            self.state.center_move_history.append(0.0)
            self.state.time_history.append(0.1)
            self.state.mean_time = 0.1

    def initialize_selection(self):
        if self.feature_importance is not None:
            df = self.feature_importance
            top_10_percent = int(len(df) * 0.1)
            top_features = df.head(top_10_percent)['Feature'].values
            self.selected_features = [int(f.replace('F', '')) for f in top_features]
        else:
            n_features = self.X.shape[1]
            self.selected_features = np.random.choice(n_features, int(n_features * 0.1), replace=False)
        self.selected_samples = self.sample_hierarchy.select_samples_from_clusters(0.1)
        self.state.sample_ratio = len(self.selected_samples) / self.X.shape[0]
        self.state.feature_ratio = len(self.selected_features) / self.X.shape[1]

    def add_samples(self):
        new_samples = self.sample_hierarchy.select_samples_from_clusters(0.1)
        if len(new_samples) > 0:
            if self.selected_samples is None:
                self.selected_samples = new_samples
            else:
                self.selected_samples = np.unique(np.concatenate([self.selected_samples, new_samples]))
        self.state.sample_ratio = self.sample_hierarchy.get_selected_ratio()
        print(f"    [DEBUG] add_samples 后: 总样本 {len(self.selected_samples)}, 比例 {self.state.sample_ratio:.3f}")

    def add_features(self, ratio):
        if self.feature_importance is not None:
            df = self.feature_importance
            all_features = [int(f.replace('F', '')) for f in df['Feature'].values]
            available = np.setdiff1d(all_features, self.selected_features)
            n_to_add = int(self.X.shape[1] * ratio)
            if len(available) > 0:
                new_features = available[:min(n_to_add, len(available))]
                self.selected_features = np.unique(np.concatenate([self.selected_features, new_features]))
        self.state.feature_ratio = len(self.selected_features) / self.X.shape[1]

    def run_clustering_feature_cdc(self):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k = len(np.unique(self.y_true))
        n_features = X_selected.shape[1]
        if hasattr(X_selected, 'toarray'):
            X_selected = X_selected.toarray()
        if n_features <= 10:
            hidden_dims = [64, 32]
            latent_dim = 10
        elif n_features <= 50:
            hidden_dims = [128, 64]
            latent_dim = 10
        else:
            hidden_dims = [256, 128, 64]
            latent_dim = 10
        from cdc import CDC
        self.cdc_model = CDC(
            input_dim=n_features, n_clusters=k, latent_dim=latent_dim,
            hidden_dims=hidden_dims, alpha=0.1, beta=0.5,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.cdc_model.pretrain(X_selected, epochs=10, batch_size=4000)
        self.cdc_model.train_epoch(X_selected, epochs=5, batch_size=4000)
        labels, centers, _ = self.cdc_model.predict(X_selected)
        return labels, centers

    def run_clustering_sample_cdc(self, new_samples=None):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        if hasattr(self, 'cdc_model') and self.cdc_model is not None:
            if self.cdc_model.input_dim == X_selected.shape[1]:
                labels, centers, _ = self.cdc_model.predict(X_selected)
                return labels, centers
        k = len(np.unique(self.y_true))
        n_features = X_selected.shape[1]
        from cdc import CDC
        if n_features <= 10:
            hidden_dims = [64, 32];
            latent_dim = 10
        elif n_features <= 50:
            hidden_dims = [128, 64];
            latent_dim = 10
        else:
            hidden_dims = [256, 128, 64];
            latent_dim = 10
        self.cdc_model = CDC(
            input_dim=n_features, n_clusters=k, latent_dim=latent_dim,
            hidden_dims=hidden_dims, alpha=0.1, beta=0.01,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.cdc_model.pretrain(X_selected, epochs=10, batch_size=4000)
        labels, centers, _ = self.cdc_model.predict(X_selected)
        return labels, centers

    def run_clustering_feature(self):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k_y = len(np.unique(self.y_true))
        mbk = MiniBatchKMeans(n_clusters=k_y, batch_size=1000)
        labels = mbk.fit_predict(X_selected)
        centers = mbk.cluster_centers_
        return labels, centers

    def run_clustering_feature(self):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k_y = len(np.unique(self.y_true))
        mbk = MiniBatchKMeans(n_clusters=k_y, batch_size=1000, n_init=3, max_iter=100, random_state=42)
        labels = mbk.fit_predict(X_selected)
        centers = mbk.cluster_centers_
        return labels, centers

    def run_clustering_sample(self, new_samples):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k_y = len(np.unique(self.y_true))
        if self.current_centers is not None and self.current_centers.shape[1] == X_selected.shape[1] and len(
                self.current_centers) == k_y:
            mbk = MiniBatchKMeans(n_clusters=k_y, init=self.current_centers, n_init=1, batch_size=10240, max_iter=100)
        else:
            mbk = MiniBatchKMeans(n_clusters=k_y, batch_size=10240, n_init=3, max_iter=100, random_state=42)

        labels = mbk.fit_predict(X_selected)
        centers = mbk.cluster_centers_
        return labels, centers

    def run_clustering_sample_idec(self, new_samples):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        if hasattr(X_selected, 'toarray'):
            X_selected = X_selected.toarray()
        k = len(np.unique(self.y_true))
        current_input_dim = X_selected.shape[1]
        if not hasattr(self, 'idec') or self.idec.input_dim != current_input_dim:
            from idec import IDEC
            self.idec = IDEC(
                input_dim=current_input_dim, n_clusters=k,
                hidden_dims=[200, 200, 1000],
                latent_dim=10,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        labels = self.idec.fit(X_selected, epochs=30, batch_size=256)
        centers = self.idec.get_cluster_centers()
        return labels, centers

    def run_clustering_feature_idec(self):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k = len(np.unique(self.y_true))
        if hasattr(X_selected, 'toarray'):
            X_selected = X_selected.toarray()
        current_input_dim = X_selected.shape[1]
        if not hasattr(self, 'idec') or self.idec.input_dim != current_input_dim:
            from idec import IDEC
            if current_input_dim > 1000:
                hidden_dims = [500, 500, 2000]
                latent_dim = 10
            else:
                hidden_dims = [256, 256, 1024]
                latent_dim = 10
            self.idec = IDEC(
                input_dim=current_input_dim,
                n_clusters=k,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        labels = self.idec.fit(X_selected, epochs=5, batch_size=512)
        centers_latent = self.idec.get_cluster_centers()
        return labels, centers_latent

    def run_clustering_feature_meanshift(self):
        from sklearn.cluster import MeanShift, estimate_bandwidth
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        X_for_fit = X_selected

        bandwidth = estimate_bandwidth(X_for_fit, quantile=0.03, n_samples=min(1000, len(X_for_fit)))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X_for_fit)

        from sklearn.metrics import pairwise_distances_argmin
        labels = pairwise_distances_argmin(X_selected, ms.cluster_centers_)
        centers = ms.cluster_centers_
        return labels, centers

    def run_clustering_sample_meanshift(self, new_samples):
        from sklearn.metrics.pairwise import pairwise_distances
        if len(new_samples) == 0 or self.current_centers is None:
            return self.current_labels, self.current_centers
        X_new = self.X[new_samples][:, self.selected_features]
        distances = pairwise_distances(X_new, self.current_centers)
        new_labels = np.argmin(distances, axis=1)
        if len(self.current_centers) > 1:
            center_dists = pairwise_distances(self.current_centers)
            np.fill_diagonal(center_dists, np.inf)
            avg_center_dist = np.mean(center_dists[center_dists < np.inf])
            threshold = avg_center_dist * 0.3
            min_distances = np.min(distances, axis=1)
            far_mask = min_distances > threshold
            if np.sum(far_mask) > 0 and np.sum(far_mask) > 5:
                X_far = X_new[far_mask]
                from sklearn.cluster import MeanShift, estimate_bandwidth
                bandwidth = estimate_bandwidth(X_far, quantile=0.03)
                if bandwidth > 0:
                    ms = MeanShift(bandwidth=bandwidth)
                    far_labels = ms.fit_predict(X_far)
                    base_id = len(self.current_centers)
                    new_labels[far_mask] = base_id + far_labels
                    self.current_centers = np.vstack([self.current_centers, ms.cluster_centers_])
        all_labels = np.concatenate([self.current_labels, new_labels])
        return all_labels, self.current_centers

    def run_clustering_feature_hierarchical(self):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k = len(np.unique(self.y_true))
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg.fit_predict(X_selected)
        centers = []
        for i in range(k):
            mask = labels == i
            if np.sum(mask) > 0:
                centers.append(X_selected[mask].mean(axis=0))
        centers = np.array(centers)
        return labels, centers

    def run_clustering_sample_hierarchical(self, new_samples):
        if len(self.selected_samples) == 0:
            return self.current_labels, self.current_centers

        X_all = self.X[self.selected_samples][:, self.selected_features]
        k = len(np.unique(self.y_true))

        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        all_labels = agg.fit_predict(X_all)
        current_centers = np.array([X_all[all_labels == i].mean(axis=0) for i in range(k)])
        return all_labels, self.current_centers

    def run_clustering_feature_gmm(self):
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k = len(np.unique(self.y_true))
        k = min(k, X_selected.shape[0] // 10) if X_selected.shape[0] > 0 else k
        k = max(k, 2)  
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
            max_iter=100,
            n_init=3,
            random_state=None,
            reg_covar=1e-6 
        )
        labels = gmm.fit_predict(X_selected)
        centers = gmm.means_  
        return labels, centers

    def run_clustering_sample_gmm(self, new_samples):
        if self.selected_samples is None or len(self.selected_samples) == 0:
            return self.current_labels, self.current_centers
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        k = len(np.unique(self.y_true))
        k = min(k, X_selected.shape[0] // 10) if X_selected.shape[0] > 0 else k
        k = max(k, 2)
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3)
        labels = gmm.fit_predict(X_selected)
        centers = gmm.means_
        return labels, centers

    def _run_pure_spectral_clustering(self):
        X_current = self.X[self.selected_samples][:, self.selected_features]
        n_samples, n_features = X_current.shape
        k = len(np.unique(self.y_true)) if hasattr(self, 'y_true') else 2
        k = min(k, n_samples - 1) if n_samples > k else k
        k = max(k, 2)
        spectral = SpectralClustering(
            n_clusters=k,
            affinity='rbf',
            gamma=1.0 / max(n_features, 1),  
            random_state=42,  
            assign_labels='kmeans',
            n_init=10
        )
        labels = spectral.fit_predict(X_current)
        centers = []
        for i in range(k):
            mask = labels == i
            if np.sum(mask) > 0:
                centers.append(X_current[mask].mean(axis=0))
            else:
                centers.append(X_current[np.random.randint(0, n_samples)])
        self.current_labels = labels
        self.current_centers = np.array(centers)
        return self.current_labels, self.current_centers
    def run_clustering_feature_spectral(self):
        return self._run_pure_spectral_clustering()
    def run_clustering_sample_spectral(self, new_samples):
        if len(self.selected_samples) == 0:
            return getattr(self, 'current_labels', []), getattr(self, 'current_centers', None)
        return self._run_pure_spectral_clustering()

    def _fill_empty_cluster_center(self, X, labels, target_cluster):
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            return X.mean(axis=0)
        from sklearn.metrics.pairwise import pairwise_distances
        non_empty_centers = []
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                non_empty_centers.append(X[mask].mean(axis=0))
        if len(non_empty_centers) == 0:
            return X.mean(axis=0)
        if target_cluster >= len(non_empty_centers):
            return non_empty_centers[0]
        return non_empty_centers[target_cluster % len(non_empty_centers)]

    def compute_state_update(self, labels, centers, time_cost):
        from sklearn.metrics import normalized_mutual_info_score
        from sklearn.neighbors import NearestNeighbors
        X_selected = self.X[self.selected_samples][:, self.selected_features]
        y_selected = self.y_true[self.selected_samples].astype(int)
        labels = labels.astype(int)
        nmi = normalized_mutual_info_score(y_selected, labels)
        self.state.nmi_history.append(nmi)

        def compute_neighbor_consistency(X, labels, k=10):
            nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
            _, indices = nbrs.kneighbors(X)
            consistency = 0
            for i in range(len(X)):
                neighbors = indices[i][1:]
                same_cluster = np.sum(labels[neighbors] == labels[i])
                consistency += same_cluster / k
            return consistency / len(X)
        consistency = compute_neighbor_consistency(X_selected, labels)
        self.state.consistency_history.append(consistency)
        if hasattr(self, 'selected_samples_prev') and self.selected_samples_prev is not None:
            common = np.intersect1d(self.selected_samples_prev, self.selected_samples, assume_unique=True)
            if len(common) > 0 and self.current_labels_prev is not None:
                prev_map = {s: i for i, s in enumerate(self.selected_samples_prev)}
                curr_map = {s: i for i, s in enumerate(self.selected_samples)}
                old_indices = [prev_map[s] for s in common if s in prev_map]
                new_indices = [curr_map[s] for s in common if s in curr_map]
                min_len = min(len(old_indices), len(new_indices))
                old_indices = old_indices[:min_len]
                new_indices = new_indices[:min_len]
                if min_len > 0:
                    old_labels = self.current_labels_prev[old_indices]
                    new_labels = labels[new_indices]
                    jaccard = compute_pairwise_jaccard(old_labels, new_labels)
                    self.state.jaccard_history.append(jaccard)
                else:
                    self.state.jaccard_history.append(0.0)
            else:
                self.state.jaccard_history.append(0.0)
        else:
            self.state.jaccard_history.append(1.0)
        self.state.time_history.append(time_cost)
        if len(self.state.time_history) > 0:
            self.state.mean_time = np.mean(list(self.state.time_history))
            self.state.std_time = max(np.std(list(self.state.time_history)), 0.01)
        self.state.iteration += 1

    def check_termination(self):
        if self.state.iteration > 5 and self.global_best_nmi > 0.1696:
            recent_nmi = list(self.state.nmi_history)[-3:]
            if len(recent_nmi) >= 10 and all(nmi > 0.5 for nmi in recent_nmi):
                return True
        if self.state.iteration >= 10:
            return True
        if self.state.sample_ratio >= 0.95 and self.state.feature_ratio >= 0.95:
            return True
        return False

    def train(self, max_episodes=10, batch_size=32, callback=None):
        total_steps = 0
        algo_elapsed_time = 0.0
        for episode in range(max_episodes):
            t_reset = time.time()
            self.reset_env()
            algo_elapsed_time += (time.time() - t_reset)
            episode_reward = 0
            step_count = 0
            while True:
                step_start_time = time.time()
                state_vec = self.state.get_state_vector()
                action = self.agent.select_action(state_vec)
                import copy
                old_state = copy.deepcopy(self.state)
                start_time = time.time()
                old_n_samples = len(self.selected_samples)
                if action == 0:
                    self.add_samples()
                    if len(self.selected_samples) > old_n_samples:
                        new_added = self.selected_samples[old_n_samples:]
                        if self.method == 'dbscan':
                            labels, centers = self.run_clustering_sample_dbscan(new_added)
                        elif self.method == 'cdc':
                            labels, centers = self.run_clustering_sample_cdc(new_added)
                        elif self.method == 'meanshift':
                            labels, centers = self.run_clustering_sample_meanshift(new_added)
                        elif self.method == 'idec':
                            labels, centers = self.run_clustering_sample_idec(new_added)
                        elif self.method == 'hierarchical':
                            labels, centers = self.run_clustering_sample_hierarchical(new_added)
                        elif self.method == 'gmm':
                            labels, centers = self.run_clustering_sample_gmm(new_added)
                        elif self.method == 'spectral':
                            labels, centers = self.run_clustering_sample_spectral(new_added)
                        else:
                            labels, centers = self.run_clustering_sample(new_added)
                    else:
                        labels, centers = self.current_labels, self.current_centers
                else:
                    self.add_features(0.1)
                    if self.method == 'dbscan':
                        labels, centers = self.run_clustering_feature_dbscan()
                    elif self.method == 'cdc':
                        labels, centers = self.run_clustering_feature_cdc()
                    elif self.method == 'meanshift':
                        labels, centers = self.run_clustering_feature_meanshift()
                    elif self.method == 'hierarchical':
                        labels, centers = self.run_clustering_feature_hierarchical()
                    elif self.method == 'gmm':
                        labels, centers = self.run_clustering_feature_gmm()
                    elif self.method == 'spectral':
                        labels, centers = self.run_clustering_feature_spectral()
                    elif self.method == 'birch':
                        labels, centers = self.run_clustering_feature_birch()
                    elif self.method == 'idec':
                        labels, centers = self.run_clustering_feature_idec()
                    else:
                        labels, centers = self.run_clustering_feature()
                time_cost = time.time() - start_time
                self.current_labels = labels
                self.current_centers = centers
                self.compute_state_update(labels, centers, time_cost)
                next_state_vec = self.state.get_state_vector()
                reward = self.reward_fn.compute_reward(old_state, self.state, action, time_cost)
                episode_reward += reward
                done = self.check_termination()
                self.agent.buffer.push(state_vec, action, reward, next_state_vec, done)
                total_steps += 1
                if len(self.agent.buffer) >= batch_size:
                    self.agent.update(batch_size=batch_size)
                curr_nmi = self.state.nmi_history[-1]
                if curr_nmi > self.global_best_nmi:
                    self.global_best_nmi = curr_nmi
                    self.global_best_samples = self.selected_samples.copy()
                    self.global_best_features = self.selected_features.copy()
                    self.global_best_labels = self.current_labels.copy()
                    self.global_best_centers = self.current_centers.copy()
                    if self.method == 'idec' and hasattr(self, 'idec') and self.idec is not None:
                        import copy
                        self.best_idec = copy.deepcopy(self.idec)
                        self.best_features = self.selected_features.copy()
                    elif self.method == 'cdc' and hasattr(self, 'cdc_model') and self.cdc_model is not None:
                        import copy
                        self.best_cdc = copy.deepcopy(self.cdc_model)
                        self.best_features = self.selected_features.copy()

                algo_elapsed_time += (time.time() - step_start_time)
                if callback is not None:
                    callback(episode, step_count, self, algo_elapsed_time)
                step_count += 1
                if done:
                    self.last_episode_features = self.selected_features.copy()
                    break
        self.state.sample_ratio = len(self.selected_samples) / self.X.shape[0]
        self.state.feature_ratio = len(self.selected_features) / self.X.shape[1]
        return self.history

if __name__ == "__main__":
    import numpy as np
    from Sample import FeatureImportancePCA, SampleHierarchy
    filename = 'MNIST.txt'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    fi = FeatureImportancePCA(n_components=10,method="pca").fit(X)
    df_importance = fi.get_importance_df([f"F{i}" for i in range(X.shape[1])])
    sh = SampleHierarchy(n_clusters=10,selection_strategy="random")

    sh.fit(X)
    sh.summary()
    rl = RLClustering(X, y_true=y, feature_importance=df_importance, sample_hierarchy=sh, method='dbscan')
    history = rl.train(max_episodes=10)
