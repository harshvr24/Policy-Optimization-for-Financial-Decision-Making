# sampling/sample_lc.py  -- updated robust version
import os
import sys
import argparse
import random
import pandas as pd
from time import time

def reservoir_sample_csv(
    csv_path: str,
    out_path: str,
    n_samples: int = 200000,
    chunksize: int = 100_000,
    random_state: int = 42,
    progress_every: int = 100_000
):
    """
    One-pass reservoir sampling that stores rows as dicts (robust for DataFrame creation).
    Saves partial reservoir on KeyboardInterrupt.
    """
    if random_state is not None:
        random.seed(random_state)

    compression = "gzip" if csv_path.lower().endswith(".gz") else None

    # Iterator of chunks
    reader = pd.read_csv(csv_path, compression=compression, chunksize=chunksize, low_memory=False)

    reservoir = []
    total_seen = 0
    t0 = time()
    print(f"[INFO] Start reservoir sampling: n={n_samples}, chunksize={chunksize}")
    print(f"[INFO] Reading from: {csv_path}")

    try:
        for chunk in reader:
            # iterate rows as dicts to avoid Series-index incompatibilities
            for row in chunk.to_dict(orient="records"):
                total_seen += 1
                if len(reservoir) < n_samples:
                    reservoir.append(row)
                else:
                    j = random.randrange(total_seen)
                    if j < n_samples:
                        reservoir[j] = row

                if total_seen % progress_every == 0:
                    elapsed = time() - t0
                    print(f"[INFO] Processed {total_seen:,} rows (elapsed {elapsed:.1f}s)")

    except KeyboardInterrupt:
        print("\n[WARNING] Sampling interrupted by user (KeyboardInterrupt).")
        # proceed to save partial reservoir below
    except Exception as e:
        print("\n[ERROR] Exception during sampling:", e)
        # still attempt to save whatever we have
    finally:
        if len(reservoir) == 0:
            print("[ERROR] No rows in reservoir to save.")
            raise SystemExit(1)

        # Convert list of dicts to DataFrame (keys will union across dicts)
        sample_df = pd.DataFrame(reservoir)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_compression = "gzip" if out_path.lower().endswith(".gz") else None

        sample_df.to_csv(out_path, index=False, compression=out_compression)
        elapsed = time() - t0
        print(f"[SUCCESS] Saved {len(sample_df):,} sampled rows -> {out_path} (scanned {total_seen:,} rows, elapsed {elapsed:.1f}s)")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Reservoir sampling for LendingClub dataset (robust)")
    parser.add_argument("--in", dest="inpath", required=True, help="Input CSV (.csv or .csv.gz)")
    parser.add_argument("--out", dest="outpath", required=True, help="Output sample path (.csv or .csv.gz)")
    parser.add_argument("--n", dest="n_samples", type=int, default=200000, help="Number of rows to sample")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for reading")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--progress-every", type=int, default=100_000, help="Print progress every X rows")

    args = parser.parse_args()

    reservoir_sample_csv(
        csv_path=args.inpath,
        out_path=args.outpath,
        n_samples=args.n_samples,
        chunksize=args.chunksize,
        random_state=args.random_state,
        progress_every=args.progress_every
    )

if __name__ == "__main__":
    main()
