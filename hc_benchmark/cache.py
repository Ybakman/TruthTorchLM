"""The Stage-A generation cache (protocol §6 shared-generation control).

One cache file per (dataset, generator, decoding, seed), holding each item's primary
answer plus up to ``n_max`` samples. Every UQ method and every N in the sweep reads this
same file, which is what makes the comparison fair: no method pays a sampling cost another
avoided, and the N-sweep is a truncation of a fixed draw rather than N fresh generations.

Stored as Parquet (columnar, typed, and it round-trips the nested sample lists), keyed by
the config's content hash so a decoding change can never be silently mixed with an old
draw.
"""

import json
from pathlib import Path

import pandas as pd

__all__ = ["GenerationCache", "CachedItem"]

CACHE_SCHEMA_VERSION = 1


class CachedItem(dict):
    """One item's cached generations. A dict subclass so it serializes transparently."""


class GenerationCache:
    """Read/write access to one (config, seed) shard of Stage-A generations."""

    def __init__(self, root: str, config, seed: int):
        self.root = Path(root)
        self.config = config
        self.seed = seed
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        h = self.config.content_hash()
        return self.root / f"stageA_{self.config.dataset}_{self.config.generator}_{h}_seed{self.seed}.parquet"

    def exists(self) -> bool:
        return self.path.exists()

    # -- write ----------------------------------------------------------

    def write(self, items: list) -> None:
        """Persist Stage-A rows.

        Each row must carry: ``item_id``, ``question``, ``context``, ``ground_truths``,
        ``primary_answer``, ``samples`` (list[str], length <= n_max), and the optional
        ``stratum`` / ``outcome_type``. Nested lists are JSON-encoded so the Parquet
        schema stays flat and portable across engines.
        """
        rows = []
        for item in items:
            row = dict(item)
            row["ground_truths"] = json.dumps(row.get("ground_truths", []))
            row["samples"] = json.dumps(row.get("samples", []))
            rows.append(row)
        df = pd.DataFrame(rows)
        df.attrs["schema_version"] = CACHE_SCHEMA_VERSION
        # Sidecar with the full config, so a cache file is self-describing on disk.
        self.path.with_suffix(".config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2)
        )
        df.to_parquet(self.path, index=False)

    # -- read -----------------------------------------------------------

    def read(self, n: int = None) -> list:
        """Load cached items, optionally truncating each item's samples to the first ``n``.

        Truncation is how the N-sweep reuses one draw: ``read(n=3)`` and ``read(n=10)``
        return prefixes of the same cached sample list, so the two budgets differ only in
        how many of the identical samples they see.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"No Stage-A cache at {self.path}. Run stage_a_generate first."
            )
        df = pd.read_parquet(self.path)
        items = []
        for _, row in df.iterrows():
            item = CachedItem(row.to_dict())
            item["ground_truths"] = json.loads(item["ground_truths"])
            samples = json.loads(item["samples"])
            if n is not None:
                if n > len(samples):
                    raise ValueError(
                        f"Requested n={n} but only {len(samples)} samples were cached for "
                        f"item {item.get('item_id')}. Regenerate Stage A with a larger n_max."
                    )
                samples = samples[:n]
            item["samples"] = samples
            items.append(item)
        return items

    def __len__(self) -> int:
        if not self.exists():
            return 0
        return len(pd.read_parquet(self.path, columns=["item_id"]))
