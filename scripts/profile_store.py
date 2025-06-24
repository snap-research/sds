import os
import time
import string
import random
import numpy as np
import pandas as pd
import polars as pl
import tiledb
import shutil

# --- Configuration ---
# NUM_TOTAL_ROWS = 100_000_000
# NUM_READ_ROWS = 1_000_000
# CHUNK_SIZE = 5_000_000
mult = 10
NUM_TOTAL_ROWS = 1_000_000 * mult
NUM_READ_ROWS = 10_000 * mult
CHUNK_SIZE = 50_000 * mult

# --- File Paths ---
HDF5_PATH = "data/large_dataset.h5"
TILEDB_PATH = "data/large_dataset.tiledb"
PARQUET_PATH = "data/large_dataset.parquet"

# --- Helper function for data generation ---
def generate_data_chunk(size):
    """Generates a chunk of data as a pandas DataFrame."""
    # Pre-generate characters for faster string creation
    chars = string.ascii_letters + string.digits

    data = {
        "name": [''.join(random.choices(chars, k=10)) for _ in range(size)],
        "id": [''.join(random.choices(chars, k=25)) for _ in range(size)],
        # For captions, generate a list of random lengths first, then the strings
        "caption": [''.join(random.choices(chars, k=random.randint(0, 100))) for _ in range(size)],
    }
    return pd.DataFrame(data)

# --- Creation Functions ---

def create_hdf5_store():
    """Creates an HDF5 store in 'table' format, which allows querying."""
    print(f"--- Creating HDF5 Store ({HDF5_PATH}) ---")
    if os.path.exists(HDF5_PATH):
        print("HDF5 file already exists. Skipping creation.")
        return

    start_time = time.perf_counter()
    for i in range(NUM_TOTAL_ROWS // CHUNK_SIZE):
        print(f"Writing chunk {i+1}/{NUM_TOTAL_ROWS // CHUNK_SIZE} to HDF5...")
        chunk_df = generate_data_chunk(CHUNK_SIZE)
        # We need to set the index to be the global row number
        chunk_df.index = pd.RangeIndex(start=i * CHUNK_SIZE, stop=(i + 1) * CHUNK_SIZE)

        chunk_df.to_hdf(
            HDF5_PATH,
            key='data',
            mode='a' if i > 0 else 'w',
            format='table',
            append=True,
            data_columns=['name', 'id', 'caption'] # Indexing columns can speed up some queries
        )
    end_time = time.perf_counter()
    print(f"HDF5 store created in {end_time - start_time:.2f} seconds.\n")

def create_tiledb_store():
    """Creates a TileDB array with a schema matching the DataFrame."""
    print(f"--- Creating TileDB Store ({TILEDB_PATH}) ---")
    if os.path.isdir(TILEDB_PATH):
        print("TileDB array already exists. Skipping creation.")
        return

    start_time = time.perf_counter()

    # Define the TileDB schema
    # The dimension is our row index.
    # FIX: The tile extent must be <= the domain size. Let's tie it to CHUNK_SIZE.
    if CHUNK_SIZE > NUM_TOTAL_ROWS:
        raise ValueError("CHUNK_SIZE cannot be greater than NUM_TOTAL_ROWS")

    dim = tiledb.Dim(name="rows", domain=(0, NUM_TOTAL_ROWS - 1), tile=CHUNK_SIZE, dtype=np.uint64)
    dom = tiledb.Domain(dim)

    # The attributes are our columns
    attrs = [
        tiledb.Attr(name="name", dtype='S10', var=False),
        tiledb.Attr(name="id", dtype='S25', var=False),
        tiledb.Attr(name="caption", dtype=str, var=True) # var=True for variable-length strings
    ]

    schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs)
    tiledb.Array.create(TILEDB_PATH, schema)

    # Write data chunk by chunk
    with tiledb.open(TILEDB_PATH, 'w') as A:
        for i in range(NUM_TOTAL_ROWS // CHUNK_SIZE):
            print(f"Writing chunk {i+1}/{NUM_TOTAL_ROWS // CHUNK_SIZE} to TileDB...")
            chunk_df = generate_data_chunk(CHUNK_SIZE)
            start_idx = i * CHUNK_SIZE
            end_idx = start_idx + CHUNK_SIZE
            A[start_idx:end_idx] = {
                'name': chunk_df['name'].values.astype('S10'),
                'id': chunk_df['id'].values.astype('S25'),
                'caption': chunk_df['caption'].values
            }

    end_time = time.perf_counter()
    print(f"TileDB store created in {end_time - start_time:.2f} seconds.\n")

def create_polars_parquet_store():
    """Creates a single Parquet file."""
    print(f"--- Creating Polars/Parquet Store ({PARQUET_PATH}) ---")
    if os.path.exists(PARQUET_PATH):
        print("Parquet file already exists. Skipping creation.")
        return

    start_time = time.perf_counter()
    # Parquet files are immutable, so appending is not ideal.
    # The common pattern is to write multiple files (a dataset).
    # For this test, we will build the full DataFrame in memory and write once.
    # Note: This step itself can cause OOM on machines with < 32GB RAM.
    # A more robust version would use pyarrow.ParquetWriter to append row groups.
    print("Generating full 100M row DataFrame in memory (this may take a while)...")

    # In a real low-memory scenario, you'd use pyarrow's ParquetWriter
    # to append row groups from chunks without holding everything in memory.
    # For this benchmark, we'll demonstrate a single-file write.
    full_df = pd.concat(
        [generate_data_chunk(CHUNK_SIZE) for i in range(NUM_TOTAL_ROWS // CHUNK_SIZE)],
        ignore_index=True
    )

    # Convert to Polars DataFrame
    pl_df = pl.from_pandas(full_df)
    # Add an index column so we can filter on it later
    pl_df = pl_df.with_row_index("index")

    print(f"Writing {NUM_TOTAL_ROWS} rows to {PARQUET_PATH}...")
    pl_df.write_parquet(PARQUET_PATH, use_pyarrow=True, row_group_size=CHUNK_SIZE)

    end_time = time.perf_counter()
    print(f"Parquet store created in {end_time - start_time:.2f} seconds.\n")


# --- Benchmark Functions ---

def benchmark_hdf5_read(indices):
    """Benchmarks reading random indices from HDF5."""
    print("--- Benchmarking HDF5 Read ---")
    start_time = time.perf_counter()

    # HDF5 (via pandas) has a 'where' clause that is perfect for this.
    # It efficiently selects rows based on the index.
    df = pd.read_hdf(HDF5_PATH, key='data', where='index in indices')

    end_time = time.perf_counter()
    print(f"Read {len(df)} rows from HDF5 in {end_time - start_time:.4f} seconds.")
    print(f"Result DataFrame shape: {df.shape}\n")
    return df

def benchmark_tiledb_read(indices):
    """Benchmarks reading random indices from TileDB."""
    print("--- Benchmarking TileDB Read ---")
    start_time = time.perf_counter()

    # TileDB's multi_index is its superpower for this exact use case.
    with tiledb.open(TILEDB_PATH, 'r') as A:
        results = A.multi_index[indices]

    # Convert result to DataFrame for apples-to-apples comparison
    df = pd.DataFrame(results)

    end_time = time.perf_counter()
    print(f"Read {len(df)} rows from TileDB in {end_time - start_time:.4f} seconds.")
    print(f"Result DataFrame shape: {df.shape}\n")
    return df

def benchmark_polars_parquet_read(indices):
    """Benchmarks reading random indices from Parquet with Polars."""
    print("--- Benchmarking Polars/Parquet Read ---")
    start_time = time.perf_counter()

    # Parquet is not optimized for random row access by index.
    # The idiomatic way is to scan the file and filter. Polars' lazy engine
    # optimizes this, but it still requires reading much more data off disk
    # than the other methods.
    # We added an 'index' column during creation to filter on.

    # Using a set for the `is_in` check is much faster for large lists
    indices_set = set(indices)

    df = pl.scan_parquet(PARQUET_PATH) \
           .filter(pl.col("index").is_in(indices_set)) \
           .collect()

    end_time = time.perf_counter()
    print(f"Read {len(df)} rows from Parquet in {end_time - start_time:.4f} seconds.")
    print(f"Result DataFrame shape: {df.shape}\n")
    return df


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Create the data stores if they don't exist
    create_hdf5_store()
    create_tiledb_store()
    # Be cautious with this one on memory-constrained systems
    create_polars_parquet_store() # Uncomment to run parquet creation
    if not os.path.exists(PARQUET_PATH):
        print("Parquet file does not exist. Please run create_polars_parquet_store() first.")
        print("Skipping Polars/Parquet benchmark.\n")

    # 2. Generate a single set of random indices for a fair comparison
    print("--- Generating Random Indices ---")
    np.random.seed(42)
    random_indices = np.random.choice(NUM_TOTAL_ROWS, size=NUM_READ_ROWS, replace=False)
    # Sorting can sometimes improve read performance, especially for disk-based formats
    random_indices.sort()
    print(f"Generated {len(random_indices)} unique random indices to fetch.\n")

    # 3. Run the benchmarks
    df_hdf5 = benchmark_hdf5_read(random_indices.tolist()) # HDF5 `where` works well with lists
    df_tiledb = benchmark_tiledb_read(random_indices)
    if os.path.exists(PARQUET_PATH):
        df_polars = benchmark_polars_parquet_read(random_indices)

    print("--- Results & Analysis ---")
    print("""
    - HDF5 (table format): Should be quite fast. The 'table' format creates B-tree indexes
      on the data, allowing for efficient lookups of specific rows without scanning the
      entire file. This is a great fit for this access pattern.

    - TileDB: Should be the fastest or competitive with HDF5. TileDB is a multi-dimensional
      array format designed from the ground up for efficient slicing and dicing. The
      `multi_index` operation is optimized for exactly this kind of random, non-contiguous
      read workload.

    - Polars (from Parquet): Will almost certainly be the slowest FOR THIS SPECIFIC TASK.
      This is not a flaw in Polars or Parquet, but a feature of its design. Parquet is a
      columnar format optimized for analytical queries (e.g., SELECT AVG(col_A) FROM table).
      It achieves this by reading whole columns or large row groups. To find 1M random rows,
      it must read large portions of the file into memory and then filter, which involves
      significant I/O overhead. This benchmark highlights the importance of matching your
      storage format to your primary access pattern.
    """)

    # 4. Clean up large files
    # print("\n--- Cleanup ---")
    # try:
    #     user_input = input("Delete the large data files? (y/N): ")
    #     if user_input.lower() == 'y':
    #         if os.path.exists(HDF5_PATH):
    #             os.remove(HDF5_PATH)
    #             print(f"Removed {HDF5_PATH}")
    #         if os.path.isdir(TILEDB_PATH):
    #             shutil.rmtree(TILEDB_PATH)
    #             print(f"Removed {TILEDB_PATH}")
    #         if os.path.exists(PARQUET_PATH):
    #             os.remove(PARQUET_PATH)
    #             print(f"Removed {PARQUET_PATH}")
    # except Exception as e:
    #     print(f"Error during cleanup: {e}")
