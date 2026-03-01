"""Graph Neural Network (GNN) pour la PD ajustee par le reseau (Contagion Supply Chain).

Ce modele utilise un Graph Attention Network (GATv2) pour capturer
la propagation du risque via la chaine d'approvisionnement.

Le principe est le suivant :
    - Nœuds : Caracteristiques financieres des entreprises.
    - Aretes : Liens 'client-fournisseur' (supply chain) ponderes.
    - Mecanisme : Le GAT apprend a accorder plus d'attention (poids)
      aux partenaires commerciaux presentant un profil de risque eleve,
      permettant de propager le stress avant meme qu'il n'impacte
      les finances locales de l'entreprise.

Ce modele agit comme un 'Challenger' ou un ajustement systémique par rapport
au modele tabulaire (XGBoost/TabNet) de reference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, MessagePassing
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import PD_CONFIG
from ifrs9_cockpit.utils.frame_compat import to_pandas


class VolumeAggregator(MessagePassing):
    """Agregation par addition pour capturer le risque de volume (petits clients multiples)."""
    def __init__(self):
        # aggr='add' est le secret : au lieu de moyenner (softmax), on accumule la detresse.
        super().__init__(aggr='add')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr a pour dimension [E, 2] : [poids, type_lien]
        weight = edge_attr[:, 0].unsqueeze(-1)
        is_client = edge_attr[:, 1].unsqueeze(-1)
        
        # On accumule proportionnellement au poids (CA).
        # On peut se concentrer sur les clients pour le volume (choc de demande).
        return x_j * weight * is_client


class GATPDModel(torch.nn.Module):
    """Architecture PyTorch Duale pour le risque de credit : Attention (Fournisseurs) + Volume (Clients)."""
    
    def __init__(self, in_channels: int, hidden_dim: int = 32, heads: int = 4, dropout: float = 0.1):
        super(GATPDModel, self).__init__()
        self.dropout = dropout
        
        # --- Voie 1 : Concentration Risk (GAT) ---
        self.gat1 = GATv2Conv(in_channels, hidden_dim, heads=heads, dropout=dropout, edge_dim=2)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=2)
        
        # --- Voie 2 : Volume Risk (Sum Aggregation) ---
        self.vol_agg = VolumeAggregator()
        # Une couche lineaire pour projeter le volume accumulé à la meme dimension
        self.vol_proj = torch.nn.Linear(in_channels, hidden_dim)
        
        # --- Couche de classification finale ---
        # On concatene les deux voies : hidden_dim (GAT) + hidden_dim (Volume)
        self.classifier = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Voie 1 : Attention ---
        x_gat = F.dropout(x, p=self.dropout, training=self.training)
        x_gat = self.gat1(x_gat, edge_index, edge_attr=edge_weight)
        x_gat = F.elu(x_gat)
        
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        x_gat = self.gat2(x_gat, edge_index, edge_attr=edge_weight)
        x_gat = F.elu(x_gat)
        
        # --- Voie 2 : Volume ---
        # Le VolumeAggregator additionne mathématiquement le stress de tous les petits clients
        x_vol = self.vol_agg(x, edge_index, edge_weight)
        x_vol = self.vol_proj(x_vol)
        x_vol = F.elu(x_vol)
        
        # --- Fusion ---
        x_combined = torch.cat([x_gat, x_vol], dim=1)
        
        # Logit prediction
        out = self.classifier(x_combined)
        return torch.sigmoid(out).squeeze(-1)


class GNNPDChallenger:
    """Encapsulation Sklearn-like pour le GNN de risque de credit.
    
    Gère la preparation des donnees tabulaires vers le format Graph (PyG)
    et orchestre l'entrainement et l'inference.
    """
    
    def __init__(
        self,
        hidden_dim: int = PD_CONFIG.gnn_hidden_dim,
        n_heads: int = PD_CONFIG.gnn_n_heads,
        dropout: float = PD_CONFIG.gnn_dropout,
        lr: float = PD_CONFIG.gnn_lr,
        epochs: int = PD_CONFIG.gnn_epochs,
        patience: int = PD_CONFIG.gnn_patience,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device)
        
        self.model: Optional[GATPDModel] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def _prepare_graph_data(
        self,
        df_nodes,
        df_links=None,
        is_train: bool = False
    ) -> Data:
        """Convertit les DataFrames en objet torch_geometric.data.Data."""
        df_nodes = to_pandas(df_nodes)
        if df_links is not None:
            df_links = to_pandas(df_links)
        
        # 1. Preparation des noeuds (features numeriques)
        # On exclut les colonnes non-predictives ou cibles
        exclude_cols = ["enterprise_id", "sector", "company_size", "loan_type", 
                        "default_flag", "pd_origination", "pd_latent"]
        
        if is_train:
            self.feature_names = [c for c in df_nodes.columns if c not in exclude_cols and df_nodes[c].dtype in ("float64", "float32", "int64", "int32")]
            X_numpy = df_nodes[self.feature_names].fillna(0).values
            X_scaled = self.scaler.fit_transform(X_numpy)
        else:
            X_numpy = df_nodes[self.feature_names].fillna(0).values
            X_scaled = self.scaler.transform(X_numpy)
            
        x_tensor = torch.tensor(X_scaled, dtype=torch.float, device=self.device)
        
        # 2. Cible (optionnelle pour l'inference)
        y_tensor = None
        if "default_flag" in df_nodes.columns:
            y_tensor = torch.tensor(df_nodes["default_flag"].values, dtype=torch.float, device=self.device)
            
        # 3. Preparation des aretes (liens supply chain)
        if df_links is not None and len(df_links) > 0:
            # Creation de mapping d'index pour s'assurer que les noeuds pointent au bon endroit
            # enterprise_id -> idx_dans_tenseur
            id_mapping = {eid: idx for idx, eid in enumerate(df_nodes["enterprise_id"].values)}
            
            # Filtrer les liens pour ne garder que ceux dont la source/cible existent dans df_nodes
            valid_links = df_links[
                df_links["source_id"].isin(id_mapping) & 
                df_links["target_id"].isin(id_mapping)
            ]
            
            # Map IDs to Tenseur Indices
            src = valid_links["source_id"].map(id_mapping).values
            dst = valid_links["target_id"].map(id_mapping).values
            weights = valid_links["weight"].values
            
            # Encodage du type de lien : 0 = fournisseur, 1 = client
            # Si la colonne n'existe pas (fallback), tout est à 0.
            types = np.zeros(len(valid_links))
            if "type" in valid_links.columns:
                types = (valid_links["type"] == "is_client_of").astype(float).values
                
            edge_features = np.vstack([weights, types]).T # Shape: [num_edges, 2]
            
            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=self.device)
            edge_weight = torch.tensor(edge_features, dtype=torch.float, device=self.device)
        else:
            # Fallback (Self-loop) : le graphe est vide, l'entreprise est liee a elle meme
            # Permet au GNN de fonctionner comme un simple MLP si aucun reseau n'est fourni.
            n_nodes = len(df_nodes)
            edge_index = torch.tensor(np.vstack([np.arange(n_nodes), np.arange(n_nodes)]), dtype=torch.long, device=self.device)
            edge_features = np.ones((n_nodes, 2)) # self-loop: weight=1, type=0
            edge_weight = torch.tensor(edge_features, dtype=torch.float, device=self.device)

        return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight, y=y_tensor)

    def fit(self, df_nodes, df_links=None) -> "GNNPDChallenger":
        """Entraine le Graph Attention Network."""
        
        graph_data = self._prepare_graph_data(df_nodes, df_links, is_train=True)
        in_channels = graph_data.x.shape[1]
        
        self.model = GATPDModel(
            in_channels=in_channels,
            hidden_dim=self.hidden_dim,
            heads=self.n_heads,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = torch.nn.BCELoss()  # Binary Cross Entropy
        
        # Desequilibre de classe
        n_pos = graph_data.y.sum().item()
        n_neg = len(graph_data.y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)

        # Early stopping logic (utilise sur le training loss car on passe tout le graphe)
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            out = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            
            # Loss ponderee (BCELoss manuelle avec poids)
            loss = F.binary_cross_entropy(out, graph_data.y, weight=torch.where(graph_data.y == 1, pos_weight, 1.0))
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
                
        self._is_fitted = True
        return self

    def predict_proba(self, df_nodes, df_links=None) -> np.ndarray:
        """Predit la PD a partir des finances (noeuds) et du reseau (arêtes)."""
        if not self._is_fitted:
            raise ValueError("Le modele GNN doit etre fitte avant prediction.")
            
        graph_data = self._prepare_graph_data(df_nodes, df_links, is_train=False)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            
        return out.cpu().numpy()


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    
    print("=" * 60)
    print("IFRS 9 COCKPIT - GNN Contagion Model Test")
    print("=" * 60)
    
    print("\\n1. Generation des donnees + Graphe Supply Chain")
    df_credit, _, _, df_links = generate_dataset(n_clients=2000, seed=42)
    print(f"  Noeuds : {len(df_credit)}")
    print(f"  Aretes : {len(df_links)}")
    
    print("\\n2. Entrainement du GATv2")
    gnn = GNNPDChallenger(epochs=50)
    gnn.fit(df_credit, df_links)
    print("  Entrainement termine.")
    
    print("\\n3. Inference sur reseau complet")
    preds_network = gnn.predict_proba(df_credit, df_links)
    print(f"  PD Moyenne (Network) : {preds_network.mean():.4f}")
    
    print("\\n4. Inference en mode Isole (Self-loop uniquement)")
    # En ne passant pas de liens, le GNN evalue l'entreprise uniquement sur ses
    # propres merites financiers, ignorant son exposition a des fournisseurs en difficulte.
    preds_isolated = gnn.predict_proba(df_credit, None)
    print(f"  PD Moyenne (Isolee)  : {preds_isolated.mean():.4f}")
    
    # Delta de contagion
    # Si le reseau contient du stress, la PD reseau sera superieure a la PD isolee
    # pour les noeuds lies a des partenaires toxiques.
    contagion_effect = preds_network - preds_isolated
    print("\\nEffet du Reseau (Contagion Delta):")
    print(f"  Delta Max : +{contagion_effect.max() * 10000:.1f} bps")
    print(f"  Delta Min : {contagion_effect.min() * 10000:.1f} bps")
    
    print("\\nGNN Module OK.")
