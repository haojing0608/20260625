# idec.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import warnings

warnings.filterwarnings('ignore')


class Autoencoder(nn.Module):
    """自编码器"""

    def __init__(self, input_dim, hidden_dims=[500, 500, 2000], latent_dim=10):
        super().__init__()

        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hdim))
            encoder_layers.append(nn.BatchNorm1d(hdim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hdim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for hdim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hdim))
            decoder_layers.append(nn.BatchNorm1d(hdim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hdim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class IDEC:
    """
    Improved Deep Embedded Clustering (IDEC)

    论文: "Improved Deep Embedded Clustering with Local Structure Preservation"
    作者: Guo et al., IJCAI 2017
    """

    def __init__(self, input_dim, n_clusters, hidden_dims=[500, 500, 2000],
                 latent_dim=10, alpha=1.0, device='cpu'):
        """
        参数:
            input_dim: 输入特征维度
            n_clusters: 聚类数
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            alpha: KL散度损失的权重系数
            device: 'cpu' 或 'cuda'
        """
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.device = device

        # 初始化模型
        self.autoencoder = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)
        self.cluster_centers = None

        # 优化器
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)

        # 训练状态
        self.is_pretrained = False
        self.y_pred = None

    def pretrain(self, X, epochs=50, batch_size=256, verbose=True):
        """
        预训练自编码器（只优化重构损失）
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"\n[IDEC] 预训练自编码器...")
        print(f"  样本数: {len(X)}, 特征维度: {self.input_dim}")
        print(f"  隐藏层: {self.hidden_dims}, 潜在维度: {self.latent_dim}")

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                self.optimizer.zero_grad()
                z, x_recon = self.autoencoder(x)
                loss = F.mse_loss(x_recon, x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.is_pretrained = True
        print(f"[IDEC] 预训练完成")

    def _initialize_cluster_centers(self, X):
        """用K-means初始化聚类中心"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z = self.autoencoder.encode(X_tensor).cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        y_pred = kmeans.fit_predict(z)
        cluster_centers = kmeans.cluster_centers_

        # 转换为tensor
        self.cluster_centers = torch.FloatTensor(cluster_centers).to(self.device)
        self.y_pred = y_pred

        return y_pred, cluster_centers

    def _compute_target_distribution(self, z):
        """
        计算目标分布 p (Student's t-distribution)

        q_ij = (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2) / Σ_k (1 + ||z_i - μ_k||^2 / α)^(-(α+1)/2)
        p_ij = q_ij^2 / Σ_k q_ik^2
        """
        # 计算距离
        dist = torch.cdist(z, self.cluster_centers)

        # Student's t-distribution
        q = 1.0 / (1.0 + dist ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)

        # 目标分布 p
        p = q ** 2 / q.sum(dim=0, keepdim=True)
        p = p / p.sum(dim=1, keepdim=True)

        return p, q

    def fit(self, X, epochs=100, batch_size=256, update_interval=10,
            tol=0.001, verbose=True):
        """
        联合训练（同时优化重构损失和聚类损失）

        参数:
            X: 输入数据 (n_samples, input_dim)
            epochs: 训练轮数
            batch_size: 批大小
            update_interval: 更新目标分布的间隔
            tol: 收敛阈值
            verbose: 是否打印信息
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 如果没有预训练，先预训练
        if not self.is_pretrained:
            self.pretrain(X, epochs=50, batch_size=batch_size, verbose=verbose)

        # 初始化聚类中心
        y_pred, cluster_centers = self._initialize_cluster_centers(X)

        print(f"\n[IDEC] 开始联合训练...")
        print(f"  聚类数: {self.n_clusters}")
        print(f"  训练轮数: {epochs}")
        print(f"  更新间隔: {update_interval}")

        best_loss = float('inf')
        y_last = y_pred.copy()

        for epoch in range(epochs):
            # 计算当前特征和聚类分配
            with torch.no_grad():
                z = self.autoencoder.encode(X_tensor)
                dist = torch.cdist(z, self.cluster_centers)
                y_pred_new = torch.argmin(dist, dim=1).cpu().numpy()

            # 计算delta_label
            delta_label = np.sum(y_pred != y_pred_new) / len(y_pred)

            # 每 update_interval 轮更新一次目标分布
            if epoch % update_interval == 0:
                with torch.no_grad():
                    z_all = self.autoencoder.encode(X_tensor)
                    p, q = self._compute_target_distribution(z_all)

                # 计算聚类损失
                log_q = torch.log(q + 1e-10)
                cluster_loss = F.kl_div(log_q, p, reduction='batchmean')

            # 训练一个epoch
            total_recon_loss = 0
            total_cluster_loss = 0
            total_loss = 0
            # ========== 修改：使用索引获取 batch 对应的 p ==========
            # 创建带索引的数据加载器
            dataset_with_idx = torch.utils.data.TensorDataset(X_tensor, torch.arange(len(X)))
            dataloader_with_idx = torch.utils.data.DataLoader(dataset_with_idx, batch_size=batch_size, shuffle=True)

            for batch_data in dataloader_with_idx:
                x, idx = batch_data  # idx 是当前batch的索引
                self.optimizer.zero_grad()

                z, x_recon = self.autoencoder(x)

                # 重构损失
                recon_loss = F.mse_loss(x_recon, x)

                # 聚类损失
                dist = torch.cdist(z, self.cluster_centers)
                q_batch = 1.0 / (1.0 + dist ** 2 / self.alpha)
                q_batch = q_batch ** ((self.alpha + 1.0) / 2.0)
                q_batch = q_batch / q_batch.sum(dim=1, keepdim=True)

                # 关键修改：使用索引获取当前batch对应的p
                p_batch = p[idx]  # 从完整的p中取出当前batch的部分

                log_q_batch = torch.log(q_batch + 1e-10)
                cluster_loss_batch = F.kl_div(log_q_batch, p_batch, reduction='batchmean')

                # 总损失 = 重构损失 + 聚类损失
                loss = recon_loss + self.alpha * cluster_loss_batch

                loss.backward()
                self.optimizer.step()

                total_recon_loss += recon_loss.item()
                total_cluster_loss += cluster_loss_batch.item()
                total_loss += loss.item()
            # ===================================================

            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_cluster_loss = total_cluster_loss / len(dataloader)
            avg_loss = total_loss / len(dataloader)

            # 更新聚类中心
            with torch.no_grad():
                z_all = self.autoencoder.encode(X_tensor)
                for k in range(self.n_clusters):
                    mask = y_pred_new == k
                    if np.sum(mask) > 0:
                        self.cluster_centers[k] = z_all[mask].mean(dim=0)

            # 更新预测标签
            y_pred = y_pred_new

            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} | "
                      f"Recon Loss: {avg_recon_loss:.4f} | "
                      f"Cluster Loss: {avg_cluster_loss:.4f} | "
                      f"Total Loss: {avg_loss:.4f} | "
                      f"Delta Label: {delta_label:.4f}")

            # 检查收敛
            if epoch > 0 and delta_label < tol:
                print(f"  [收敛] Delta Label = {delta_label:.4f} < {tol}")
                break

        self.y_pred = y_pred
        print(f"[IDEC] 训练完成")

        return y_pred

    def predict(self, X):
        """预测聚类标签"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z = self.autoencoder.encode(X_tensor)
            dist = torch.cdist(z, self.cluster_centers)
            labels = torch.argmin(dist, dim=1).cpu().numpy()
        return labels

    def get_cluster_centers(self):
        """获取聚类中心"""
        return self.cluster_centers.cpu().numpy()

    def get_embedding(self, X):
        """获取潜在空间表示"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z = self.autoencoder.encode(X_tensor)
        return z.cpu().numpy()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'cluster_centers': self.cluster_centers,
            'input_dim': self.input_dim,
            'n_clusters': self.n_clusters,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'alpha': self.alpha
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.cluster_centers = checkpoint['cluster_centers'].to(self.device)
        self.input_dim = checkpoint['input_dim']
        self.n_clusters = checkpoint['n_clusters']
        self.hidden_dims = checkpoint['hidden_dims']
        self.latent_dim = checkpoint['latent_dim']
        self.alpha = checkpoint['alpha']
        self.is_pretrained = True


# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(1000, 100)
    y_true = np.random.randint(0, 10, 1000)

    # 创建IDEC模型
    idec = IDEC(
        input_dim=100,
        n_clusters=10,
        hidden_dims=[256, 128],
        latent_dim=50,
        alpha=1.0
    )

    # 训练
    y_pred = idec.fit(X, epochs=50, batch_size=128)

    # 评估
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"\nNMI: {nmi:.4f}")