# lau-tile-compress

**Tile compression — store more by storing less.** A lossy compression library that preserves meaning, not bytes. Designed for tile-based data (grids, maps, sensor readings) where perfect fidelity is optional but structural integrity matters.

Six compression strategies, a pluggable pipeline, and a semantic merge engine that groups similar tiles by cosine similarity.

## What This Does

You have tiles — grids of `f64` values — and you want to compress them. This library gives you:

- **Run-Length Encoding (RLE):** Repeated values collapse into (count, value) pairs. Perfect for uniform regions.
- **Delta Encoding:** Store the first value + differences. Ideal for slowly-changing data (heightmaps, temperature grids).
- **Dictionary Encoding:** Repeated patterns of 4 values get mapped to 16-bit codes. First occurrence is literal; subsequent ones are 2 bytes.
- **Threshold Filtering:** Zero out small values based on a quality parameter (0.0 = aggressive, 1.0 = lossless).
- **Semantic Compression:** Group tiles with high cosine similarity and merge them into a single averaged tile.
- **Hybrid:** Automatically picks whichever of RLE or delta produces smaller output.

A `CompressionPipeline` chains multiple stages sequentially, and `CompressionStats` tracks ratios and savings.

## Key Idea

Not all data deserves lossless compression. Tile-based game maps, sensor grids, and AI observation tensors often contain redundancy at the *semantic* level — two tiles might be "close enough" to merge without anyone noticing.

This library treats compression as a spectrum:
- **Quality = 1.0:** Lossless. Every byte preserved.
- **Quality = 0.0:** Aggressive. Small values zeroed, similar tiles merged, maximum space savings.
- **Quality = 0.5–0.9:** Practical. Good compression with controlled information loss.

The **delta encoder** also computes **Shannon entropy** of the deltas, which tells you how compressible the data is before you even choose a strategy.

## Install

```toml
[dependencies]
lau-tile-compress = "0.1.0"
```

## Quick Start

```rust
use lau_tile_compress::*;

// Create a tile with repetitive data
let tile = RawTile::new("tile-42", "room-7", vec![1.0; 100]);

// Compress with RLE at quality 1.0 (lossless)
let compressor = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
let compressed = compressor.compress(&tile);
println!("Compressed {} → {} bytes", tile.byte_size(), compressed.data.len());

// Decompress back
let restored = compressor.decompress(&compressed);
assert_eq!(restored.content, tile.content);

// Check stats
let stats = compressor.stats(&[tile], &[compressed]);
println!("Ratio: {:.1}x, Savings: {:.0}%", stats.ratio(), stats.savings_percent());
```

### Semantic compression of multiple tiles

```rust
let tiles = vec![
    RawTile::new("a", "room", vec![1.0, 2.0, 3.0, 4.0]),
    RawTile::new("b", "room", vec![1.01, 2.02, 3.01, 4.01]),  // very similar
    RawTile::new("c", "room", vec![100.0, 200.0, 300.0, 400.0]), // different
];

let semantic = SemanticCompressor::new(0.99); // 99% similarity threshold
let compressed = semantic.compress(&tiles);
// "a" and "b" merged; "c" stays separate
```

### Pipeline: chain stages

```rust
let mut pipeline = CompressionPipeline::new();
pipeline.add_stage(Box::new(CompressorStage::new(
    TileCompressor::new(CompressionStrategy::Threshold, 0.8)
)));
pipeline.add_stage(Box::new(CompressorStage::new(
    TileCompressor::new(CompressionStrategy::RunLength, 1.0)
)));

let tile = RawTile::new("t1", "room", my_data);
let result = pipeline.compress(&tile);
```

## API Reference

### `RawTile`

An uncompressed tile with content, metadata, and size estimation.

| Field / Method | Description |
|---|---|
| `id` | Tile identifier |
| `room_id` | Parent room/region |
| `timestamp` | Creation time |
| `content` | The `Vec<f64>` data |
| `metadata` | Key-value string metadata |
| `new(id, room_id, content)` | Constructor |
| `byte_size()` | Approximate in-memory size |

### `CompressedTile`

The result of compression.

| Field / Method | Description |
|---|---|
| `id` | Original tile ID |
| `strategy` | Which strategy was used |
| `data` | Compressed byte payload |
| `original_size` | Size before compression |
| `quality` | Quality parameter used |
| `is_lossless()` | True if quality == 1.0 |

### `CompressionStrategy`

```rust
enum CompressionStrategy {
    RunLength,    // (count, value) pairs
    Dictionary,   // pattern → u16 code
    Delta,        // first value + differences
    Threshold,    // zero out small values
    Semantic,     // merge similar tiles
    Hybrid,       // pick best of RLE vs delta
}
```

### `TileCompressor`

The main engine. Selects strategy and quality.

| Method | Description |
|---|---|
| `new(strategy, quality)` | Create with strategy and quality ∈ [0, 1] |
| `compress(&tile)` | Compress a single tile |
| `decompress(&compressed)` | Decompress back to `RawTile` |
| `compress_batch(&[tiles])` | Compress multiple tiles |
| `stats(&original, &compressed)` | Compute `CompressionStats` |

### `RunLengthEncoder`

Encodes runs of identical `f64` values. Format: `[u32 LE count][f64 LE value]` per run (12 bytes per run).

| Method | Description |
|---|---|
| `new()` | Create encoder |
| `encode(&[f64])` | Encode to bytes |
| `decode(&[u8])` | Decode back to `Vec<f64>` |

### `DictionaryEncoder`

Maps patterns of 4 `f64` values to `u16` codes. First occurrence stored as literal (33 bytes); subsequent ones as 3 bytes (flag + code).

| Method | Description |
|---|---|
| `new()` | Create with empty dictionary |
| `encode(&mut self, &[f64])` | Encode (mutates dictionary) |
| `decode(&self, &[u8])` | Decode using current dictionary |
| `dictionary_size()` | Number of learned patterns |

### `DeltaEncoder`

Stores first value + successive differences.

| Method | Description |
|---|---|
| `new()` | Create encoder |
| `encode(&[f64])` | Convert to deltas |
| `decode(first, &deltas)` | Reconstruct from first value + deltas |
| `delta_entropy(&[f64])` | Shannon entropy of deltas (measures compressibility) |

### `SemanticCompressor`

Groups tiles by cosine similarity and merges similar ones into averaged tiles.

| Method | Description |
|---|---|
| `new(threshold)` | Create with similarity threshold ∈ [0, 1] |
| `similarity(&a, &b)` | Cosine similarity between two vectors |
| `merge(&[&RawTile])` | Weighted average of tile contents |
| `compress(&[RawTile])` | Group and merge, returning `Vec<CompressedTile>` |

### `CompressionPipeline`

Chains multiple `TileCompressorTrait` stages.

| Method | Description |
|---|---|
| `new()` | Create empty pipeline |
| `add_stage(Box<dyn TileCompressorTrait>)` | Append a stage |
| `compress(&mut self, &tile)` | Run all stages sequentially |
| `total_ratio()` | Cumulative compression ratio |

### `CompressionStats`

| Method | Description |
|---|---|
| `new(original, compressed)` | Create stats |
| `ratio()` | original / compressed (∞ if compressed = 0) |
| `savings_percent()` | Percentage of space saved |
| `is_worthwhile()` | True if ratio > 1.5× |

### `CompressorStage`

Wraps a `TileCompressor` to implement `TileCompressorTrait` for pipeline use.

```rust
let stage = CompressorStage::new(TileCompressor::new(CompressionStrategy::RunLength, 1.0));
pipeline.add_stage(Box::new(stage));
```

## How It Works

### Run-Length Encoding

Scans the `f64` array linearly. When the current value matches the previous (within `f64::EPSILON`), increment the run count. On change, emit `[count: u32 LE][value: f64 LE]` (12 bytes). Best for uniform or slowly-varying data.

A 1000-element array of all `3.14` compresses to exactly 12 bytes (one run). A strictly alternating `[1.0, 2.0, 1.0, 2.0, ...]` compresses to 24 bytes per pair — worse than raw.

### Delta Encoding

Transforms `[x₀, x₁, x₂, ...]` → `[x₀, x₁−x₀, x₂−x₁, ...]`. For smoothly varying data, deltas are small numbers that compress well with any subsequent scheme. Reconstruction is cumulative: `xᵢ = x₀ + Σ(dⱼ)`.

The encoder also computes **Shannon entropy** of quantized deltas (bin size 0.1):

```
H = −Σ pᵢ · ln(pᵢ)
```

Low entropy (e.g., monotonic data) = highly compressible. High entropy (random data) = not much to gain.

### Dictionary Encoding

Chunks the input into groups of 4 `f64` values. Each unique chunk gets a `u16` code. First occurrence is stored as a literal (flag byte + 4×8 bytes = 33 bytes). Subsequent occurrences use only 3 bytes (flag byte + 2-byte code). Effective when the same patterns repeat.

The dictionary persists across calls to `encode()`, so encoding multiple tiles with the same encoder accumulates shared patterns.

### Threshold Filtering

Not an encoder itself — a preprocessing step. Given quality `q ∈ [0, 1]`, computes:

```
threshold = (1 − q) × max(|content|)
```

Any value with `|v| < threshold` is zeroed. At quality 1.0, nothing is removed (lossless). At quality 0.0, everything except the maximum absolute value is zeroed.

### Semantic Compression

Groups tiles by cosine similarity. For each unassigned tile, scans remaining tiles and groups those with similarity ≥ threshold. Each group is merged into a single tile whose content is the element-wise average:

```
merged[i] = (1/n) × Σ group_tile[i]
```

The resulting `CompressedTile` stores the averaged content as JSON with quality 0.5 (always lossy). The merged tile has lower fidelity but captures the common structure of the group.

### Hybrid Strategy

Runs both RLE and delta on the same data, then picks whichever produces fewer bytes. Adds one compression pass of overhead but guarantees you don't accidentally pick the worse strategy.

### Pipeline

A `CompressionPipeline` chains `TileCompressorTrait` implementors. Input flows through stage 1 (compress → decompress → recompress through stage 2 → ...). Each stage applies its own strategy. For example: threshold → RLE first zeros small values, then RLE exploits the resulting runs of zeros.

## The Math

### Shannon Entropy

For a discrete random variable with probabilities pᵢ:

```
H(X) = −Σᵢ pᵢ log(pᵢ)
```

The delta encoder estimates this by:
1. Computing deltas: `dᵢ = xᵢ − xᵢ₋₁`
2. Quantizing into bins of size 0.1: `bin = round(d / 0.1)`
3. Computing frequency-based probabilities: `pᵢ = count(binᵢ) / total`
4. Computing the sum

Result is in **nats** (natural units, using ln). For a constant signal, entropy = 0. For uniform random data, entropy approaches log(n) where n is the number of distinct bins.

### Cosine Similarity

Used by the semantic compressor to measure how "aligned" two tile contents are:

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖) = Σ(aᵢbᵢ) / (√Σ(aᵢ²) × √Σ(bᵢ²))
```

Range: [−1, 1]. 1.0 = identical direction, 0.0 = orthogonal, −1.0 = opposite. The compressor only merges tiles with similarity ≥ threshold (typically 0.9–0.99).

### Compression Ratio

```
ratio = original_bytes / compressed_bytes
savings = (1 − compressed_bytes / original_bytes) × 100%
```

Ratio > 1.0 means compression is working. The library considers compression "worthwhile" when ratio > 1.5×.

### Delta Entropy and Compressibility

Shannon's source coding theorem tells us the best achievable compression ratio is bounded by the entropy H of the source. For delta-encoded data with entropy H nats per symbol:

- **Minimum bits per delta** ≈ H / ln(2)
- **Theoretical max ratio** = (original_bits_per_value × n) / (H / ln(2) × n)

Low entropy deltas (smooth signals) compress extremely well; high entropy deltas (noisy signals) don't.

## Testing

57 integration tests covering:

- RLE: roundtrip, constant data, random data, ratio bounds
- Delta: roundtrip, monotonic data, constant data, entropy calculation
- Dictionary: repeated patterns, roundtrip, size growth
- Semantic: similarity measures, merging, edge cases (empty, zero vectors)
- Pipeline: sequential stages, empty pipeline, default construction
- Quality: lossless at 1.0, aggressive at 0.0, partial filtering
- Batch: compress multiple tiles, stats accuracy
- Serde: roundtrip serialization for all major types
- Edge cases: empty tiles, single-element tiles, default constructors

Run with `cargo test`.

## License

MIT
