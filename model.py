
from typing import Dict, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPE(nn.Module):
    """Hybrid Peptide Encoder.

    Encodes amino-acid token ids and precomputed ESM residue features into a
    shared peptide representation.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        token_dim: int = 128,
        esm_dim: int = 320,
        n_filters: int = 128,
        filter_sizes: List[int] = None,
        dropout: float = 0.6,
        pool_size: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        self.token_dim = token_dim
        self.esm_dim = esm_dim
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        input_dim = token_dim + esm_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=n_filters, kernel_size=k, padding="same")
            for k in filter_sizes
        ])
        flat_dim = len(filter_sizes) * n_filters * pool_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, len(filter_sizes) * n_filters * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(len(filter_sizes) * n_filters * 2, hidden_dim),
            nn.Mish(),
        )

    def _features(self, sequence_ids: torch.Tensor, esm_features: torch.Tensor) -> torch.Tensor:
        token_embedding = self.embedding(sequence_ids.long())
        length = min(esm_features.size(1), token_embedding.size(1))
        return torch.cat(
            [esm_features[:, :length, :].float(), token_embedding[:, :length, :]],
            dim=-1,
        )

    def forward(self, sequence_ids: torch.Tensor, esm_features: torch.Tensor) -> torch.Tensor:
        peptide_features = self._features(sequence_ids, esm_features)
        x = peptide_features.permute(0, 2, 1)
        conv_outputs = [self.act(conv(x)) for conv in self.convs]
        pooled_outputs = [F.adaptive_max_pool1d(output, self.pool_size) for output in conv_outputs]
        flat_features = torch.cat([pooled.flatten(start_dim=1) for pooled in pooled_outputs], dim=1)
        return self.mlp(self.dropout(flat_features))


class PredictionBranch(nn.Module):
    """Prediction branch used inside BPN."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Mish(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.norm(representation + self.net(representation))


class BPN(nn.Module):
    """Base Prediction Network.

    Produces the primary multi-label logits and probe-based uncertainty
    descriptors used by the base prediction path.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_size: int = 21,
        expert_dropout: float = 0.3,
        candidate_threshold: float = 0.5,
    ):
        super().__init__()
        self.output_size = output_size
        self.candidate_threshold = candidate_threshold

        self.probe_predictor = nn.Linear(hidden_dim, output_size)
        self.uncertainty_summarizer = nn.Sequential(
            nn.Linear(3, 16),
            nn.Mish(),
            nn.Linear(16, 1),
        )

        self.branch1 = PredictionBranch(hidden_dim, dropout=expert_dropout)
        self.branch2 = PredictionBranch(hidden_dim, dropout=expert_dropout)
        self.uncertainty_fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 1, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(expert_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.base_classifier = nn.Linear(hidden_dim, output_size)

    @staticmethod
    def build_uncertainty_descriptor(
        probe_probabilities: torch.Tensor,
        candidate_threshold: float = 0.5,
    ) -> torch.Tensor:
        eps = 1e-8
        num_labels = probe_probabilities.size(1)
        entropy = -(
            probe_probabilities * torch.log(probe_probabilities + eps)
            + (1.0 - probe_probabilities) * torch.log(1.0 - probe_probabilities + eps)
        ).mean(dim=1)
        entropy = entropy / math.log(2.0)

        top2 = torch.topk(probe_probabilities, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        inverse_margin = 1.0 - margin

        candidate_ratio = (probe_probabilities >= candidate_threshold).float().sum(dim=1) / float(num_labels)
        uncertainty_descriptor = torch.stack([entropy, inverse_margin, candidate_ratio], dim=1)
        return uncertainty_descriptor.clamp(0.0, 1.0)

    def probe_uncertainty(self, peptide_representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        probe_logits = self.probe_predictor(peptide_representation)
        probe_probabilities = torch.sigmoid(probe_logits)
        uncertainty_descriptor = self.build_uncertainty_descriptor(
            probe_probabilities,
            self.candidate_threshold,
        )
        uncertainty_score = torch.sigmoid(self.uncertainty_summarizer(uncertainty_descriptor)).squeeze(-1)
        return {
            "probe_logits": probe_logits,
            "uncertainty_descriptor": uncertainty_descriptor,
            "uncertainty_score": uncertainty_score,
        }

    def base_from_uncertainty(
        self,
        peptide_representation: torch.Tensor,
        probe_logits: torch.Tensor,
        uncertainty_descriptor: torch.Tensor,
        uncertainty_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        branch1_representation = self.branch1(peptide_representation)
        branch2_representation = self.branch2(peptide_representation)
        uncertainty_fusion_input = torch.cat(
            [peptide_representation, uncertainty_descriptor, uncertainty_score.unsqueeze(1)],
            dim=1,
        )
        uncertainty_gate = torch.sigmoid(self.uncertainty_fusion_gate(uncertainty_fusion_input)).squeeze(-1)
        fused_representation = (
            (1.0 - uncertainty_gate).unsqueeze(1) * branch1_representation
            + uncertainty_gate.unsqueeze(1) * branch2_representation
        )
        base_logits = self.base_classifier(fused_representation)
        return {
            "base_logits": base_logits,
            "probe_logits": probe_logits,
            "uncertainty_descriptor": uncertainty_descriptor,
            "uncertainty_score": uncertainty_score,
            "uncertainty_gate": uncertainty_gate,
            "branch1_representation": branch1_representation,
            "branch2_representation": branch2_representation,
            "fused_representation": fused_representation,
        }

    def forward(self, peptide_representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        uncertainty_output = self.probe_uncertainty(peptide_representation)
        return self.base_from_uncertainty(peptide_representation, **uncertainty_output)


class CSE(nn.Module):
    """Cardinality Score Estimator.

    Estimates the coarse label-cardinality score:
        q = P(k >= 2 | x)
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_logits(self, peptide_representation: torch.Tensor) -> torch.Tensor:
        return self.head(peptide_representation).squeeze(-1)

    def forward(self, peptide_representation: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(peptide_representation))


class RCH(nn.Module):
    """Residual Completion Head."""

    def __init__(self, hidden_dim: int = 256, output_size: int = 21, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, branch2_representation: torch.Tensor) -> torch.Tensor:
        return self.head(branch2_representation)


class CGRF(nn.Module):
    """Cardinality-Guided Residual Fusion."""

    def __init__(self, lambda_res: float = 0.0):
        super().__init__()
        self.lambda_res = float(lambda_res)

    def forward(
        self,
        base_logits: torch.Tensor,
        residual_logits: torch.Tensor,
        cardinality_score: torch.Tensor,
    ) -> torch.Tensor:
        residual_gate = cardinality_score.detach().unsqueeze(1)
        return base_logits + self.lambda_res * residual_gate * residual_logits


class RLCM(nn.Module):
    """Residual Label Completion Module."""

    def __init__(self, hidden_dim: int = 256, output_size: int = 21, dropout: float = 0.3, lambda_res: float = 0.0):
        super().__init__()
        self.rch = RCH(hidden_dim=hidden_dim, output_size=output_size, dropout=dropout)
        self.cgrf = CGRF(lambda_res=lambda_res)

    def forward(
        self,
        base_logits: torch.Tensor,
        branch2_representation: torch.Tensor,
        cardinality_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        residual_logits = self.rch(branch2_representation)
        final_logits = self.cgrf(
            base_logits=base_logits,
            residual_logits=residual_logits,
            cardinality_score=cardinality_score,
        )
        return {
            "final_logits": final_logits,
            "residual_logits": residual_logits,
        }



class CRLC(nn.Module):
    """Cardinality-Guided Residual Label Completion model.

    Default mode:
        final_logits = base_logits

    Optional residual mode:
        final_logits = base_logits + lambda_res * cardinality_score.detach() * residual_logits
    """

    def __init__(
        self,
        vocab_size: int = 21,
        output_size: int = 21,
        token_dim: int = 128,
        esm_dim: int = 320,
        n_filters: int = 128,
        filter_sizes: List[int] = None,
        hidden_dim: int = 256,
        dropout: float = 0.6,
        pool_size: int = 10,
        expert_dropout: float = 0.3,
        candidate_threshold: float = 0.5,
        enable_residual: bool = False,
        lambda_res: float = 0.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.enable_residual = bool(enable_residual)
        self.hpe = HPE(
            vocab_size=vocab_size,
            token_dim=token_dim,
            esm_dim=esm_dim,
            n_filters=n_filters,
            filter_sizes=filter_sizes,
            dropout=dropout,
            pool_size=pool_size,
            hidden_dim=hidden_dim,
        )
        self.bpn = BPN(
            hidden_dim=hidden_dim,
            output_size=output_size,
            expert_dropout=expert_dropout,
            candidate_threshold=candidate_threshold,
        )
        if self.enable_residual:
            self.cse = CSE(hidden_dim=hidden_dim, dropout=expert_dropout)
            self.rlcm = RLCM(
                hidden_dim=hidden_dim,
                output_size=output_size,
                dropout=expert_dropout,
                lambda_res=lambda_res,
            )

    def forward(self, sequence_ids: torch.Tensor, esm_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        peptide_representation = self.hpe(sequence_ids, esm_features)

        if self.enable_residual:
            uncertainty_output = self.bpn.probe_uncertainty(peptide_representation)
            cardinality_logit = self.cse.forward_logits(peptide_representation)
            cardinality_score = torch.sigmoid(cardinality_logit)
            base_output = self.bpn.base_from_uncertainty(
                peptide_representation,
                **uncertainty_output,
            )
        else:
            base_output = self.bpn(peptide_representation)

        base_logits = base_output["base_logits"]
        residual_logits = torch.zeros_like(base_logits)
        final_logits = base_logits

        if self.enable_residual:
            residual_output = self.rlcm(
                base_logits=base_logits,
                branch2_representation=base_output["branch2_representation"],
                cardinality_score=cardinality_score,
            )
            final_logits = residual_output["final_logits"]
            residual_logits = residual_output["residual_logits"]

        result = {
            "final_logits": final_logits,
            "base_logits": base_logits,
            "residual_logits": residual_logits,
            "probe_logits": base_output["probe_logits"],
            "uncertainty_descriptor": base_output["uncertainty_descriptor"],
            "uncertainty_score": base_output["uncertainty_score"],
            "uncertainty_gate": base_output["uncertainty_gate"],
            "branch1_representation": base_output["branch1_representation"],
            "branch2_representation": base_output["branch2_representation"],
            "fused_representation": base_output["fused_representation"],
        }
        if self.enable_residual:
            result["cardinality_score"] = cardinality_score
            result["cardinality_logit"] = cardinality_logit
        return result


def build_model(config: Dict, esm_dim: int, enable_residual: bool) -> CRLC:
    model_config = config["model"]
    return CRLC(
        vocab_size=21,
        output_size=21,
        token_dim=model_config["token_dim"],
        esm_dim=esm_dim,
        n_filters=model_config["n_filters"],
        filter_sizes=model_config["filter_sizes"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        pool_size=model_config.get("pool_size", 10),
        expert_dropout=model_config["expert_dropout"],
        candidate_threshold=model_config["candidate_threshold"],
        enable_residual=enable_residual,
        lambda_res=config["residual_training"]["lambda_res"] if enable_residual else 0.0,
    )
