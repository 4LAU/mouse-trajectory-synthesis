"""WS7b event-stream model: speed + heading increment.

Same event-stream premise as WS7 (generate the physical (dt, displacement)
event sequence, let the standard 125Hz resample create the human recording
artifacts), but motion is parameterized as speed and a heading INCREMENT so
directional smoothness is a property of the representation instead of
something two independent dx/dy heads must learn (WS7's failure: zigzag,
RF OOB 0.945).

Heads, all on the shared CANDI-style trunk:
- speed: categorical over [tick, 129 log-speed bins, PAD]. Only this head
  carries PAD, so decode truncation has a single owner (WS7's spurious-PAD
  clipping came from truncating on either of two heads).
- dtheta: categorical over TH_BINS uniform bins centred on 0 (the 45-degree
  lattice of 1px moves falls on exact bin centres when 8 | TH_BINS). The
  head is conditioned on the speed class at the same position, because human
  large turns happen almost only at low speed: p(s, th | ctx) =
  p(s | ctx) p(th | s, ctx). Ticks and PADs carry a NULL dtheta token and
  no dtheta loss; heading persists through them.
- dt: flow matching on z-scored log(dt_ms), unchanged from WS7.

Decode contract (experiments/event_stream_polar.py): heading starts at the
conditioning angle plus the first motion event's dtheta, positions are the
cumulative sum of s*(cos, sin) ROUNDED TO INTEGER PIXELS. The replay gate
showed off-grid positions alone are worth ~0.05 AUC to the detector
(eval_polar_b*.log); rounding restores the representation floor of ~0.51.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.candi import CANDIBlock, SinusoidalEmbedding, cosine_beta_schedule

TH_BINS = 256                     # dtheta bins per full turn, 8 | TH_BINS
S_BINS = 128                      # log-speed bins over [1, S_MAX]
S_MAX = 90.0                      # hypot(63, 63) = 89.1
S_LOG_W = math.log(S_MAX) / S_BINS

# Speed classes: 0 = tick (s == 0), 1 + k for log bin k in [0, S_BINS], PAD.
N_S_VALS = S_BINS + 1
TICK_CLASS = 0
S_PAD_CLASS = 1 + N_S_VALS        # 130
N_S_CLASSES = S_PAD_CLASS + 1     # 131
S_MASK_TOKEN = N_S_CLASSES        # input-only absorbing state

# dtheta classes: bin b in [0, TH_BINS) centred on angle (b - TH_BINS/2)*w,
# plus NULL for tick/PAD positions. NULL is input context, never a target.
TH_NULL_CLASS = TH_BINS           # 256
N_TH_CLASSES = TH_BINS + 1        # 257
TH_MASK_TOKEN = N_TH_CLASSES      # input-only absorbing state

DTH_LATTICE = 32768               # prepare_polar_events.py int16 lattice


def s2_to_class(s2: torch.Tensor) -> torch.Tensor:
    """Squared speed (int) -> speed class. 0 -> tick, else 1 + log bin."""
    s2f = s2.float().clamp(min=1.0)
    k = torch.round(0.5 * torch.log(s2f) / S_LOG_W).long().clamp(0, S_BINS)
    return torch.where(s2 > 0, k + 1, torch.zeros_like(k))


def class_to_speed(c: torch.Tensor) -> torch.Tensor:
    """Speed class -> speed value. Tick and PAD map to 0."""
    s = torch.exp((c - 1).float() * S_LOG_W)
    return torch.where((c > TICK_CLASS) & (c < S_PAD_CLASS), s, torch.zeros_like(s))


def dth_lattice_to_class(dth: torch.Tensor) -> torch.Tensor:
    """int16 lattice value in [-16384, 16383] -> bin in [0, TH_BINS)."""
    step = DTH_LATTICE // TH_BINS
    b = torch.round(dth.float() / step).long()
    return (b + TH_BINS) % TH_BINS


def class_to_dtheta(c: torch.Tensor) -> torch.Tensor:
    """Bin -> angle in radians, in (-pi, pi]. NULL maps to 0."""
    b = torch.where(c >= TH_BINS // 2, c - TH_BINS, c)
    ang = b.float() * (2.0 * math.pi / TH_BINS)
    return torch.where(c >= TH_NULL_CLASS, torch.zeros_like(ang), ang)


class EventStreamPolarModel(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,
        n_diffusion_steps: int = 1000,
        cond_dropout: float = 0.1,
        dropout: float = 0.1,
        feat_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_steps = n_diffusion_steps
        self.cond_dropout = cond_dropout
        self.feat_dim = feat_dim

        self.s_embed = nn.Embedding(N_S_CLASSES + 1, d_model)    # +1 MASK
        self.th_embed = nn.Embedding(N_TH_CLASSES + 1, d_model)  # +1 MASK
        self.dt_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # optional global "movement character" conditioning: a per-trajectory
        # feature vector (the detector's own statistics) projected into the
        # same additive slot as cond. The last layer is zeroed after init so
        # a checkpoint fine-tuned from feat_dim=0 starts as an exact no-op.
        self.feat_embed = None
        if feat_dim > 0:
            self.feat_embed = nn.Sequential(
                nn.Linear(feat_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.layers = nn.ModuleList([
            CANDIBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dt_head = nn.Linear(d_model, 1)
        self.s_head = nn.Linear(d_model, N_S_CLASSES)
        # dtheta head sees the trunk feature plus an embedding of the speed
        # class at the same position: p(th | s, ctx).
        self.s_ctx_embed = nn.Embedding(N_S_CLASSES, d_model)
        self.th_norm = nn.LayerNorm(d_model)
        self.th_head = nn.Linear(d_model, TH_BINS)

        betas = cosine_beta_schedule(n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", torch.sqrt(alpha_bar))
        self._init_weights()
        if self.feat_embed is not None:
            nn.init.zeros_(self.feat_embed[2].weight)
            nn.init.zeros_(self.feat_embed[2].bias)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def q_flow(self, x0: torch.Tensor, t_cont: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        t = t_cont.view(-1, 1)
        x_t = (1 - t) * x0 + t * noise
        velocity = noise - x0
        return x_t, noise, velocity

    def q_mask_joint(self, s_tok: torch.Tensor, th_tok: torch.Tensor, t_int: torch.Tensor):
        """Absorbing-state masking, one mask per POSITION: s and dtheta at a
        position are hidden and revealed together."""
        mask_prob = 1.0 - self.sqrt_ab[t_int].view(-1, 1)
        mask = torch.rand(s_tok.shape, device=s_tok.device) < mask_prob
        s_out = s_tok.clone()
        th_out = th_tok.clone()
        s_out[mask] = S_MASK_TOKEN
        th_out[mask] = TH_MASK_TOKEN
        return s_out, th_out, mask

    def trunk(self, dt_noisy, s_tok, th_tok, t, cond, feat=None):
        B, T = dt_noisy.shape
        x = (
            self.dt_proj(dt_noisy.unsqueeze(-1))
            + self.s_embed(s_tok)
            + self.th_embed(th_tok)
            + self.pos_embed(torch.arange(T, device=dt_noisy.device))
        )
        t_emb = self.time_embed(t)
        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(B, 1, device=cond.device) > self.cond_dropout).float()
            cond = cond * keep
        combined = t_emb + self.cond_embed(cond)
        if self.feat_embed is not None and feat is not None:
            if self.training and self.cond_dropout > 0:
                fkeep = (torch.rand(B, 1, device=feat.device) > self.cond_dropout).float()
                feat = feat * fkeep
            combined = combined + self.feat_embed(feat)
        for layer in self.layers:
            x = layer(x, combined)
        return self.norm(x)

    def th_logits(self, x: torch.Tensor, s_cond: torch.Tensor) -> torch.Tensor:
        """dtheta logits given trunk features and the speed CLASS at each
        position (true class in training, sampled class when sampling)."""
        return self.th_head(self.th_norm(x + self.s_ctx_embed(s_cond.clamp(max=N_S_CLASSES - 1))))

    def forward(self, dt_noisy, s_tok, th_tok, t, cond, s_true, feat=None):
        x = self.trunk(dt_noisy, s_tok, th_tok, t, cond, feat)
        return (
            self.dt_head(x).squeeze(-1),
            self.s_head(x),
            self.th_logits(x, s_true),
        )

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        seq_len: int,
        n_steps: int = 100,
        temperature: float = 1.0,
        th_temperature: float | None = None,
        order: str = "conf",
        choice_temp: float = 0.0,
        feat: torch.Tensor | None = None,
    ):
        """Flow ODE on dt; MaskGIT position reveal on (s, dtheta) jointly.

        Tokens are SAMPLED from the softmax (argmax telescopes, see WS7),
        the reveal count follows the training mask schedule sqrt_ab[t], and
        a position's confidence is p(s) * p(th | s) (p(s) alone at ticks).
        Reveal order controls macro path shape: pure confidence order locks
        in near-zero dtheta early (paths far too straight, path_efficiency
        0.994 vs human 0.949), pure random order wanders (0.861). order=
        "gumbel" interpolates: score = log conf + choice_temp * (1 - progress)
        * Gumbel noise, the standard MaskGIT choice-temperature anneal.
        Returns (dt_z, s_cls, th_cls): (B, T).
        """
        B = cond.shape[0]
        dev = cond.device
        temp = max(temperature, 1e-4)
        th_temp = max(th_temperature if th_temperature is not None else temperature, 1e-4)

        dt_z = torch.randn(B, seq_len, device=dev)
        s_tok = torch.full((B, seq_len), S_MASK_TOKEN, dtype=torch.long, device=dev)
        th_tok = torch.full((B, seq_len), TH_MASK_TOKEN, dtype=torch.long, device=dev)

        step = 1.0 / n_steps

        def sample_masked(masked):
            """Sample (s, th, joint confidence) at every position; only
            entries where `masked` is True are ever used."""
            s_probs = torch.softmax(s_logits / temp, dim=-1)
            s_new = torch.multinomial(
                s_probs.view(-1, s_probs.shape[-1]), 1
            ).view(B, seq_len)
            s_for_th = torch.where(masked, s_new, s_tok.clamp(max=N_S_CLASSES - 1))
            th_l = self.th_logits(x_feat, s_for_th)
            th_probs = torch.softmax(th_l / th_temp, dim=-1)
            th_new = torch.multinomial(
                th_probs.view(-1, th_probs.shape[-1]), 1
            ).view(B, seq_len)

            conf = s_probs.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
            motion = (s_new > TICK_CLASS) & (s_new < S_PAD_CLASS)
            th_conf = th_probs.gather(-1, th_new.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(motion, conf * th_conf, conf)
            th_new = torch.where(motion, th_new, torch.full_like(th_new, TH_NULL_CLASS))
            return s_new, th_new, conf

        for i in range(n_steps):
            t_cont = 1.0 - i * step
            t_scaled = torch.full((B,), t_cont * (self.n_steps - 1), device=dev)
            x_feat = self.trunk(dt_z, s_tok, th_tok, t_scaled, cond, feat)
            v_pred = self.dt_head(x_feat).squeeze(-1)
            s_logits = self.s_head(x_feat)

            dt_z = dt_z - step * v_pred

            t_next = max(t_cont - step, 0.0)
            n_target = int(round(
                float(self.sqrt_ab[int(t_next * (self.n_steps - 1))]) * seq_len
            ))
            masked = s_tok == S_MASK_TOKEN
            n_new = n_target - int(seq_len - masked[0].sum().item())
            if n_new <= 0:
                continue

            s_new, th_new, conf = sample_masked(masked)
            if order == "random":
                score = torch.rand_like(conf)
            elif order == "gumbel":
                g = -torch.log(-torch.log(torch.rand_like(conf).clamp(1e-9, 1.0)))
                anneal = choice_temp * (1.0 - i / n_steps)
                score = torch.log(conf.clamp(min=1e-9)) + anneal * g
            else:
                score = conf
            score = torch.where(masked, score, torch.full_like(score, -1e9))
            rank = score.argsort(dim=-1, descending=True)
            reveal = torch.zeros_like(masked)
            reveal.scatter_(1, rank[:, :n_new], True)
            reveal &= masked
            s_tok[reveal] = s_new[reveal]
            th_tok[reveal] = th_new[reveal]

        still = s_tok == S_MASK_TOKEN
        if still.any():
            s_new, th_new, _ = sample_masked(still)
            s_tok[still] = s_new[still]
            th_tok[still] = th_new[still]

        return dt_z, s_tok, th_tok
