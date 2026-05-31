use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 1. CompressionStats
// ---------------------------------------------------------------------------

/// Statistics about a compression operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
}

impl CompressionStats {
    pub fn new(original_bytes: usize, compressed_bytes: usize) -> Self {
        Self {
            original_bytes,
            compressed_bytes,
        }
    }

    /// Compression ratio: original / compressed. Higher is better.
    pub fn ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            if self.original_bytes == 0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            self.original_bytes as f64 / self.compressed_bytes as f64
        }
    }

    /// Percentage of space saved.
    pub fn savings_percent(&self) -> f64 {
        if self.original_bytes == 0 {
            0.0
        } else {
            (1.0 - self.compressed_bytes as f64 / self.original_bytes as f64) * 100.0
        }
    }

    /// Whether the compression ratio exceeds 1.5×.
    pub fn is_worthwhile(&self) -> bool {
        self.ratio() > 1.5
    }
}

// ---------------------------------------------------------------------------
// 3. CompressionStrategy
// ---------------------------------------------------------------------------

/// Available compression strategies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    RunLength,
    Dictionary,
    Delta,
    Threshold,
    Semantic,
    Hybrid,
}

// ---------------------------------------------------------------------------
// 4. RawTile
// ---------------------------------------------------------------------------

/// An uncompressed tile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTile {
    pub id: String,
    pub room_id: String,
    pub timestamp: u64,
    pub content: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

impl RawTile {
    pub fn new(id: impl Into<String>, room_id: impl Into<String>, content: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            room_id: room_id.into(),
            timestamp: 0,
            content,
            metadata: HashMap::new(),
        }
    }

    /// Approximate in-memory size in bytes.
    pub fn byte_size(&self) -> usize {
        let base = self.id.len()
            + self.room_id.len()
            + std::mem::size_of::<u64>() // timestamp
            + std::mem::size_of::<usize>(); // vec len overhead
        let content = self.content.len() * std::mem::size_of::<f64>();
        let meta: usize = self
            .metadata
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum();
        base + content + meta
    }
}

// ---------------------------------------------------------------------------
// 5. CompressedTile
// ---------------------------------------------------------------------------

/// A compressed tile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTile {
    pub id: String,
    pub strategy: CompressionStrategy,
    pub data: Vec<u8>,
    pub original_size: usize,
    pub quality: f64,
}

impl CompressedTile {
    pub fn is_lossless(&self) -> bool {
        (self.quality - 1.0).abs() < f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// 11. TileCompressorTrait
// ---------------------------------------------------------------------------

/// Trait for pluggable compression stages.
pub trait TileCompressorTrait: Send + Sync {
    fn compress(&self, tile: &RawTile) -> CompressedTile;
    fn decompress(&self, compressed: &CompressedTile) -> RawTile;
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// 6. RunLengthEncoder
// ---------------------------------------------------------------------------

/// Run-length encoder for `f64` slices.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunLengthEncoder;

impl RunLengthEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Encode `data` using RLE. Format: count (u32 LE) + value bytes (8 bytes LE f64) per run.
    pub fn encode(&self, data: &[f64]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(data.len() * 12);
        let mut count: u32 = 1;
        let mut current = data[0];
        for &val in &data[1..] {
            if (val - current).abs() < f64::EPSILON {
                count += 1;
            } else {
                out.extend_from_slice(&count.to_le_bytes());
                out.extend_from_slice(&current.to_le_bytes());
                current = val;
                count = 1;
            }
        }
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&current.to_le_bytes());
        out
    }

    /// Decode RLE bytes back to `f64` values.
    pub fn decode(&self, data: &[u8]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + 12 <= data.len() {
            let count = u32::from_le_bytes(
                data[pos..pos + 4].try_into().expect("count bytes"),
            );
            let val = f64::from_le_bytes(
                data[pos + 4..pos + 12].try_into().expect("value bytes"),
            );
            for _ in 0..count {
                out.push(val);
            }
            pos += 12;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 7. DictionaryEncoder
// ---------------------------------------------------------------------------

/// Dictionary-based encoder: maps patterns of 4 f64 values to 16-bit codes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEncoder {
    pub dictionary: HashMap<String, u16>,
    pub next_code: u16,
}

impl DictionaryEncoder {
    pub fn new() -> Self {
        Self {
            dictionary: HashMap::new(),
            next_code: 0,
        }
    }

    /// Encode using dictionary. Each chunk of up to 4 f64s → one u16 code if known, else literal.
    /// Format per chunk: 1 byte flag (0=literal, 1=dict) + data.
    /// Literal: 4 × f64 (32 bytes). Dict: u16 code (2 bytes).
    pub fn encode(&mut self, data: &[f64]) -> Vec<u8> {
        let mut out = Vec::new();
        let chunks = data.chunks(4);
        for chunk in chunks {
            let key: String = chunk.iter().flat_map(|v| v.to_le_bytes()).map(|b| b as char).collect();
            if let Some(&code) = self.dictionary.get(&key) {
                out.push(1u8); // dictionary flag
                out.extend_from_slice(&code.to_le_bytes());
            } else {
                // add to dictionary if space
                if self.next_code < u16::MAX {
                    self.dictionary.insert(key.clone(), self.next_code);
                    self.next_code += 1;
                }
                out.push(0u8); // literal flag
                for &v in chunk {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                // pad to 4 values if short chunk
                for _ in 0..(4 - chunk.len()) {
                    out.extend_from_slice(&0.0f64.to_le_bytes());
                }
            }
        }
        out
    }

    /// Decode dictionary-encoded data. Note: must use a populated encoder to decode.
    pub fn decode(&self, data: &[u8]) -> Vec<f64> {
        let reverse: HashMap<u16, Vec<f64>> = self
            .dictionary
            .iter()
            .filter_map(|(key, &code)| {
                let bytes: Vec<u8> = key.chars().map(|c| c as u8).collect();
                if !bytes.len().is_multiple_of(8) {
                    return None;
                }
                let vals: Vec<f64> = bytes
                    .chunks(8)
                    .map(|c| f64::from_le_bytes(c.try_into().expect("f64 bytes")))
                    .collect();
                Some((code, vals))
            })
            .collect();

        let mut out = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            let flag = data[pos];
            pos += 1;
            if flag == 1 && pos + 2 <= data.len() {
                let code = u16::from_le_bytes([data[pos], data[pos + 1]]);
                pos += 2;
                if let Some(vals) = reverse.get(&code) {
                    out.extend_from_slice(vals);
                }
            } else if pos + 32 <= data.len() {
                for _ in 0..4 {
                    let v = f64::from_le_bytes(
                        data[pos..pos + 8].try_into().expect("f64 bytes"),
                    );
                    out.push(v);
                    pos += 8;
                }
            } else {
                break;
            }
        }
        out
    }

    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }
}

impl Default for DictionaryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 8. DeltaEncoder
// ---------------------------------------------------------------------------

/// Delta encoder: stores first value + differences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaEncoder;

impl DeltaEncoder {
    pub fn new() -> Self {
        Self
    }

    pub fn encode(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut deltas = Vec::with_capacity(data.len());
        deltas.push(data[0]);
        for i in 1..data.len() {
            deltas.push(data[i] - data[i - 1]);
        }
        deltas
    }

    pub fn decode(&self, first: f64, deltas: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(deltas.len() + 1);
        let mut current = first;
        out.push(current);
        for &d in deltas {
            current += d;
            out.push(current);
        }
        out
    }

    /// Shannon entropy of the deltas.
    pub fn delta_entropy(&self, data: &[f64]) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }
        let deltas: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
        // Quantize deltas into bins for probability estimation
        let bin_size = 0.1;
        let mut freq: HashMap<i64, usize> = HashMap::new();
        for &d in &deltas {
            let bin = (d / bin_size).round() as i64;
            *freq.entry(bin).or_insert(0) += 1;
        }
        let total = deltas.len() as f64;
        let mut entropy = 0.0;
        for &count in freq.values() {
            let p = count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

// ---------------------------------------------------------------------------
// 9. SemanticCompressor
// ---------------------------------------------------------------------------

/// Compresses by merging semantically similar tiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCompressor {
    pub similarity_threshold: f64,
}

impl SemanticCompressor {
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
        }
    }

    /// Cosine similarity between two vectors.
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a < f64::EPSILON || mag_b < f64::EPSILON {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Weighted average of tile contents.
    pub fn merge(&self, tiles: &[&RawTile]) -> Vec<f64> {
        if tiles.is_empty() {
            return Vec::new();
        }
        let len = tiles[0].content.len();
        let n = tiles.len() as f64;
        let mut merged = vec![0.0; len];
        for tile in tiles {
            for (i, &v) in tile.content.iter().enumerate() {
                if i < len {
                    merged[i] += v / n;
                }
            }
        }
        merged
    }

    /// Compress by merging similar tiles into fewer outputs.
    pub fn compress(&self, tiles: &[RawTile]) -> Vec<CompressedTile> {
        if tiles.is_empty() {
            return Vec::new();
        }

        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; tiles.len()];

        for i in 0..tiles.len() {
            if assigned[i] {
                continue;
            }
            let mut group = vec![i];
            assigned[i] = true;
            for j in (i + 1)..tiles.len() {
                if assigned[j] {
                    continue;
                }
                if self.similarity(&tiles[i].content, &tiles[j].content) >= self.similarity_threshold {
                    group.push(j);
                    assigned[j] = true;
                }
            }
            groups.push(group);
        }

        groups
            .into_iter()
            .map(|group| {
                let refs: Vec<&RawTile> = group.iter().map(|&i| &tiles[i]).collect();
                let merged = self.merge(&refs);
                let tile_id = tiles[group[0]].id.clone();
                let data = serde_json::to_vec(&merged).unwrap_or_default();
                CompressedTile {
                    id: tile_id,
                    strategy: CompressionStrategy::Semantic,
                    data,
                    original_size: refs.iter().map(|t| t.byte_size()).sum(),
                    quality: 0.5,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 2. TileCompressor
// ---------------------------------------------------------------------------

/// The main compression engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileCompressor {
    pub strategy: CompressionStrategy,
    pub quality: f64,
}

impl TileCompressor {
    pub fn new(strategy: CompressionStrategy, quality: f64) -> Self {
        Self { strategy, quality }
    }

    fn apply_threshold(&self, data: &[f64]) -> Vec<f64> {
        if self.quality >= 1.0 {
            return data.to_vec();
        }
        let threshold = (1.0 - self.quality) * data.iter().cloned().fold(0.0_f64, f64::max).abs();
        data.iter()
            .map(|&v| if v.abs() < threshold { 0.0 } else { v })
            .collect()
    }

    pub fn compress(&self, tile: &RawTile) -> CompressedTile {
        let processed = self.apply_threshold(&tile.content);
        let data = match &self.strategy {
            CompressionStrategy::RunLength => {
                let enc = RunLengthEncoder::new();
                enc.encode(&processed)
            }
            CompressionStrategy::Delta => {
                let enc = DeltaEncoder::new();
                let deltas = enc.encode(&processed);
                serde_json::to_vec(&deltas).unwrap_or_default()
            }
            CompressionStrategy::Dictionary => {
                let mut enc = DictionaryEncoder::new();
                enc.encode(&processed)
            }
            CompressionStrategy::Threshold => {
                serde_json::to_vec(&processed).unwrap_or_default()
            }
            CompressionStrategy::Semantic => {
                serde_json::to_vec(&processed).unwrap_or_default()
            }
            CompressionStrategy::Hybrid => {
                let rle = RunLengthEncoder::new();
                let first_pass = rle.encode(&processed);
                let delta = DeltaEncoder::new();
                let deltas = delta.encode(&processed);
                let delta_bytes = serde_json::to_vec(&deltas).unwrap_or_default();
                if first_pass.len() < delta_bytes.len() {
                    first_pass
                } else {
                    delta_bytes
                }
            }
        };
        CompressedTile {
            id: tile.id.clone(),
            strategy: self.strategy.clone(),
            data,
            original_size: tile.byte_size(),
            quality: self.quality,
        }
    }

    pub fn decompress(&self, compressed: &CompressedTile) -> RawTile {
        let content = match &self.strategy {
            CompressionStrategy::RunLength => {
                let enc = RunLengthEncoder::new();
                enc.decode(&compressed.data)
            }
            CompressionStrategy::Delta => {
                let deltas: Vec<f64> =
                    serde_json::from_slice(&compressed.data).unwrap_or_default();
                let enc = DeltaEncoder::new();
                if deltas.is_empty() {
                    deltas
                } else {
                    enc.decode(deltas[0], &deltas[1..])
                }
            }
            CompressionStrategy::Dictionary => {
                let enc = DictionaryEncoder::new();
                enc.decode(&compressed.data)
            }
            CompressionStrategy::Threshold | CompressionStrategy::Semantic => {
                serde_json::from_slice(&compressed.data).unwrap_or_default()
            }
            CompressionStrategy::Hybrid => {
                // Try RLE first, then delta
                let rle = RunLengthEncoder::new();
                let rle_result = rle.decode(&compressed.data);
                if !rle_result.is_empty() {
                    rle_result
                } else {
                    serde_json::from_slice(&compressed.data).unwrap_or_default()
                }
            }
        };
        RawTile {
            id: compressed.id.clone(),
            room_id: String::new(),
            timestamp: 0,
            content,
            metadata: HashMap::new(),
        }
    }

    pub fn compress_batch(&self, tiles: &[RawTile]) -> Vec<CompressedTile> {
        tiles.iter().map(|t| self.compress(t)).collect()
    }

    pub fn stats(&self, original: &[RawTile], compressed: &[CompressedTile]) -> CompressionStats {
        let original_bytes: usize = original.iter().map(|t| t.byte_size()).sum();
        let compressed_bytes: usize = compressed.iter().map(|t| t.data.len()).sum();
        CompressionStats::new(original_bytes, compressed_bytes)
    }
}

// ---------------------------------------------------------------------------
// 10. CompressionPipeline
// ---------------------------------------------------------------------------

/// Chains multiple compressors sequentially.
pub struct CompressionPipeline {
    pub stages: Vec<Box<dyn TileCompressorTrait>>,
    pub cumulative_ratio: f64,
}

impl CompressionPipeline {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            cumulative_ratio: 1.0,
        }
    }

    pub fn add_stage(&mut self, compressor: Box<dyn TileCompressorTrait>) {
        self.stages.push(compressor);
    }

    pub fn compress(&mut self, tile: &RawTile) -> CompressedTile {
        if self.stages.is_empty() {
            return CompressedTile {
                id: tile.id.clone(),
                strategy: CompressionStrategy::Hybrid,
                data: serde_json::to_vec(&tile.content).unwrap_or_default(),
                original_size: tile.byte_size(),
                quality: 1.0,
            };
        }
        let mut current = tile.clone();
        let mut result = self.stages[0].compress(&current);
        for stage in &self.stages[1..] {
            current = stage.decompress(&result);
            result = stage.compress(&current);
        }
        // Update cumulative ratio
        if result.original_size > 0 {
            self.cumulative_ratio = result.original_size as f64 / result.data.len().max(1) as f64;
        }
        result
    }

    pub fn total_ratio(&self) -> f64 {
        self.cumulative_ratio
    }
}

impl Default for CompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Stage wrapper for TileCompressor to implement TileCompressorTrait
// ---------------------------------------------------------------------------

pub struct CompressorStage {
    compressor: TileCompressor,
}

impl CompressorStage {
    pub fn new(compressor: TileCompressor) -> Self {
        Self { compressor }
    }
}

impl TileCompressorTrait for CompressorStage {
    fn compress(&self, tile: &RawTile) -> CompressedTile {
        self.compressor.compress(tile)
    }

    fn decompress(&self, compressed: &CompressedTile) -> RawTile {
        self.compressor.decompress(compressed)
    }

    fn name(&self) -> &str {
        match self.compressor.strategy {
            CompressionStrategy::RunLength => "RunLength",
            CompressionStrategy::Dictionary => "Dictionary",
            CompressionStrategy::Delta => "Delta",
            CompressionStrategy::Threshold => "Threshold",
            CompressionStrategy::Semantic => "Semantic",
            CompressionStrategy::Hybrid => "Hybrid",
        }
    }
}
