"""B3 — GRU на PyTorch: последние 4 ч по 15 минут → q10, q50, q90 на 4 получасовых слота вперёд (в z).

**Вход** — момент t, вестибюль:
- 16 шагов, четверти [t − 4 ч, t); на шаге 4 канала: z четверти log((y + 1)/(норма + 1)), флаг пропуска, z всей линии
  (Σy / Σнорма по валидным вестибюлям), флаг пропуска линии. Пропуск — ночь, флаги часа, b4 < 20, нет профиля или s_v.
  z стандартизуется по обучению, пропуск после стандартизации — 0;
- эмбеддинги: вестибюль (8), номер четверти суток t (8), день недели суток метро (3), тип дня (2). Уровни — фиксированные
  списки (вестибюли сетки, 96 четвертей, 7 дней, 4 типа дня), от обучения не зависят.

**Почему GRU, а не 1D-CNN.**
- Окно короткое (16 шагов), поэтому скорость не важна.
- Главное в данных — затухание отклонения: r держится 2–3 часа, последние четверти весят больше всего (EDA, этап 6).
  Рекуррентное состояние учит это затухание напрямую.
- Пропуски (ночь перед утренними моментами t, закрытия) GRU обходит через флаги и ворота. Свёртка с фиксированными
  смещениями смешивает их с данными на тех же позициях.
- На CPU GRU детерминирована.

**Сеть.** GRU (1 слой, hidden 32) → последнее состояние + эмбеддинги → Linear 64, ReLU, dropout 0,1 → 12 выходов.
На каждый горизонт: q50, q10 = q50 − softplus, q90 = q50 + softplus — квантили не пересекаются.

**Потери** — pinball по 4 горизонтам × 3 квантилям с весом n_slot / среднее; горизонты без цели маскируются.

**Обучение.**
- AdamW, lr 1e-3, weight decay 1e-4, батч 512, до 100 эпох, терпение 8.
- Ранняя остановка — по последним 7 суткам обучения: отложенная сеть учится на остальном, стандартизация — по нему же.
  Затем дообучение на всём окне с тем же числом эпох, стандартизация — по всему окну (как число деревьев у LightGBM).
- seed 0, 1, 2, z усредняются: среднее упорядоченных троек упорядочено.
- CPU, torch.use_deterministic_algorithms, один поток. Один поток нужен потому, что в одном процессе с lightgbm две
  копии libomp и многопоточный torch падает (см. src/nn_model.py). При тех же данных и seed прогнозы совпадают побитово.
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import eda_spb as E
from src.stack import data as D

QUANTILES = (0.1, 0.5, 0.9)
DAY_TYPES = E.DAY_TYPES
EMBED = {"vestibule": 8, "qod": 8, "dow": 3, "day_type": 2}
PARAMS = {"hidden": 32, "head": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch": 512,
          "max_epochs": 100, "patience": 8, "es_days": 7, "seeds": [0, 1, 2], "threads": 1}


# --- Данные ---------------------------------------------------------------------------
@dataclass
class Samples:
    """Моменты прогноза (вестибюль, t): сырые окна, категории, цели по 4 горизонтам."""
    keys: pd.DataFrame                       # vestibule_id, t, sday
    x: np.ndarray                            # [N, 16, 4]: z, пропуск, z линии, пропуск линии (z ещё не стандартизован)
    codes: np.ndarray                        # [N, 4]: вестибюль, четверть суток, день недели, тип дня
    z: np.ndarray                            # [N, 4] цель; NaN — нет строки
    w: np.ndarray                            # [N, 4] вес n_slot; 0 — нет строки

    def subset(self, m: np.ndarray) -> "Samples":
        return Samples(self.keys[m].reset_index(drop=True), self.x[m], self.codes[m], self.z[m], self.w[m])


def samples(q: D.Quarters, rows: pd.DataFrame) -> Samples:
    """Окна и цели для всех (вестибюль, t) из rows. Окна — по меткам времени (Quarters.windows), строки rows уже
    прошли window_ok: окно не пересекает разрыв между блоками."""
    keys = (rows[["vestibule_id", "t", "sday", "day_type"]].drop_duplicates(["vestibule_id", "t"])
            .sort_values(["t", "vestibule_id"], kind="stable").reset_index(drop=True))
    if not q.window_ok(keys.t).all():
        raise ValueError("окно 4 ч выходит за блок месяца")
    _, idx = q.windows(keys.t)
    vi = pd.Index(q.vestibules).get_indexer(keys.vestibule_id)
    zq, zl = q.z(), q.z_line()
    zv = zq[idx, vi[:, None]]
    zline = zl[idx]
    x = np.stack([zv, np.isnan(zv), zline, np.isnan(zline)], axis=-1).astype(np.float32)   # NaN z — до стандартизации
    local = keys.t.dt.tz_convert(config.SPB_TZ)
    codes = np.column_stack([vi, (local.dt.hour * 4 + local.dt.minute // 15).to_numpy(),
                             keys.sday.dt.dayofweek.to_numpy(),
                             pd.Categorical(keys.day_type, categories=DAY_TYPES).codes]).astype(np.int64)
    if (codes < 0).any():
        raise ValueError("неизвестный уровень категории")
    pos = pd.MultiIndex.from_frame(keys[["vestibule_id", "t"]]).get_indexer(
        pd.MultiIndex.from_frame(rows[["vestibule_id", "t"]]))
    z = np.full((len(keys), D.K), np.nan)
    w = np.zeros((len(keys), D.K))
    z[pos, rows.k.to_numpy() - 1] = rows.z.to_numpy(float)
    w[pos, rows.k.to_numpy() - 1] = rows.n.to_numpy(float)
    return Samples(keys[["vestibule_id", "t", "sday"]], x, codes, z, w)


@dataclass
class Scaler:
    """Стандартизация каналов z и z линии — по валидным шагам обучающих окон."""
    mean: list
    std: list

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        out_m, out_s = [], []
        for c in (0, 2):
            v = x[..., c][~np.isnan(x[..., c])].astype(float)
            out_m.append(float(v.mean()))
            out_s.append(float(v.std()) or 1.0)
        return cls(out_m, out_s)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for j, c in enumerate((0, 2)):
            x[..., c] = np.nan_to_num((x[..., c] - self.mean[j]) / self.std[j], nan=0.0)
        return x.astype(np.float32)


# --- Сеть -------------------------------------------------------------------------------
def _net(n_vestibules: int, params: dict):
    import torch
    from torch import nn

    levels = {"vestibule": n_vestibules, "qod": 96, "dow": 7, "day_type": len(DAY_TYPES)}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(4, params["hidden"], batch_first=True)
            self.emb = nn.ModuleList([nn.Embedding(levels[c], EMBED[c]) for c in EMBED])
            width = params["hidden"] + sum(EMBED.values())
            self.head = nn.Sequential(nn.Linear(width, params["head"]), nn.ReLU(), nn.Dropout(params["dropout"]),
                                      nn.Linear(params["head"], D.K * 3))

        def forward(self, x, codes):
            _, h = self.gru(x)
            parts = [h[-1]] + [e(codes[:, i]) for i, e in enumerate(self.emb)]
            out = self.head(torch.cat(parts, dim=1)).view(-1, D.K, 3)
            mid = out[..., 0]
            lo = mid - nn.functional.softplus(out[..., 1])
            hi = mid + nn.functional.softplus(out[..., 2])
            return torch.stack([lo, mid, hi], dim=-1)          # [N, 4, 3]

    return Net()


def pinball(pred, z, w):
    """Взвешенный pinball: pred [N, 4, 3], z и w [N, 4] (w = 0 — нет цели)."""
    import torch
    q = torch.tensor(QUANTILES, dtype=pred.dtype)
    diff = z[..., None] - pred
    loss = torch.maximum(q * diff, (q - 1) * diff).sum(-1)
    return (loss * w).sum() / w.sum()


@dataclass
class GRUQuantile:
    n_vestibules: int
    params: dict = field(default_factory=lambda: copy.deepcopy(PARAMS))
    scaler: Scaler | None = None
    nets: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # --- служебное
    def _setup(self, seed: int):
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.params["threads"])
        return torch.Generator().manual_seed(seed)

    def _tensors(self, s: Samples, scaler: Scaler, with_target: bool = True):
        import torch
        out = [torch.from_numpy(scaler.transform(s.x)), torch.from_numpy(s.codes)]
        if with_target:
            w = s.w / s.w[s.w > 0].mean()
            out += [torch.from_numpy(np.nan_to_num(s.z).astype(np.float32)), torch.from_numpy(w.astype(np.float32))]
        return out

    def _epoch(self, net, opt, data, gen) -> None:
        import torch
        x, codes, z, w = data
        net.train()
        perm = torch.randperm(len(z), generator=gen)
        for i in range(0, len(z), self.params["batch"]):
            b = perm[i:i + self.params["batch"]]
            opt.zero_grad()
            pinball(net(x[b], codes[b]), z[b], w[b]).backward()
            opt.step()

    def _loss(self, net, data) -> float:
        import torch
        x, codes, z, w = data
        net.eval()
        with torch.no_grad():
            return float(pinball(net(x, codes), z, w))

    def _train(self, data, seed: int, epochs: int | None = None, valid=None):
        """Одна сеть: ранняя остановка по valid (epochs=None) или ровно epochs эпох."""
        import torch
        gen = self._setup(seed)
        net = _net(self.n_vestibules, self.params)
        opt = torch.optim.AdamW(net.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        if epochs is not None:
            for _ in range(epochs):
                self._epoch(net, opt, data, gen)
            return net, epochs, None
        best, best_state, best_ep, bad, curve = np.inf, None, 0, 0, []
        for ep in range(1, self.params["max_epochs"] + 1):
            self._epoch(net, opt, data, gen)
            loss = self._loss(net, valid)
            curve.append(loss)
            if loss < best - 1e-7:
                best, best_state, best_ep, bad = loss, copy.deepcopy(net.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= self.params["patience"]:
                    break
        net.load_state_dict(best_state)
        return net, best_ep, curve

    # --- интерфейс
    def split(self, s: Samples) -> tuple[np.ndarray, np.ndarray]:
        """Отложенная часть и последние es_days суток обучения (ранняя остановка)."""
        last = s.keys.sday.max()
        es = (s.keys.sday > last - pd.Timedelta(days=self.params["es_days"])).to_numpy()
        return ~es, es

    def fit(self, s: Samples) -> "GRUQuantile":
        hold_m, es_m = self.split(s)
        hold, es = s.subset(hold_m), s.subset(es_m)
        sc_hold = Scaler.fit(hold.x)
        data, valid = self._tensors(hold, sc_hold), self._tensors(es, sc_hold)
        epochs, losses = [], []
        for seed in self.params["seeds"]:
            _, ep, curve = self._train(data, seed, valid=valid)
            epochs.append(ep)
            losses.append(min(curve))
        self.scaler = Scaler.fit(s.x)
        full = self._tensors(s, self.scaler)
        self.nets = [self._train(full, seed, epochs=ep)[0] for seed, ep in zip(self.params["seeds"], epochs)]
        self.info = {"n_train": len(s.keys), "n_hold": int(hold_m.sum()), "n_es": int(es_m.sum()),
                     "train_start": str(s.keys.sday.min().date()), "train_end": str(s.keys.sday.max().date()),
                     "es_start": str(es.keys.sday.min().date()), "epochs": epochs, "es_loss": losses}
        return self

    def predict(self, s: Samples) -> np.ndarray:
        """z [N, 4, 3] — среднее по seed."""
        import torch
        x, codes = self._tensors(s, self.scaler, with_target=False)
        out = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                out.append(net(x, codes).numpy().astype(float))
        return np.mean(out, axis=0)

    def save(self, path: Path) -> None:
        import torch
        path.mkdir(parents=True, exist_ok=True)
        for seed, net in zip(self.params["seeds"], self.nets):
            torch.save(net.state_dict(), path / f"gru_seed{seed}.pt")
        (path / "meta.json").write_text(json.dumps({"params": self.params, "scaler": self.scaler.__dict__,
                                                    "n_vestibules": self.n_vestibules, "info": self.info},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GRUQuantile":
        import torch
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        m = cls(meta["n_vestibules"], meta["params"], Scaler(**meta["scaler"]), info=meta["info"])
        for seed in m.params["seeds"]:
            net = _net(m.n_vestibules, m.params)
            net.load_state_dict(torch.load(path / f"gru_seed{seed}.pt"))
            m.nets.append(net)
        return m


def to_rows(s: Samples, z: np.ndarray, rows: pd.DataFrame) -> pd.DataFrame:
    """z [N, 4, 3] по моментам → колонки b3_10, b3_50, b3_90 строк rows (вестибюль, t, k)."""
    pos = pd.MultiIndex.from_frame(s.keys[["vestibule_id", "t"]]).get_indexer(
        pd.MultiIndex.from_frame(rows[["vestibule_id", "t"]]))
    if (pos < 0).any():
        raise ValueError("нет прогноза GRU для части строк")
    k = rows.k.to_numpy() - 1
    return pd.DataFrame({f"b3_{q}": z[pos, k, j] for j, q in enumerate(("10", "50", "90"))}, index=rows.index)
