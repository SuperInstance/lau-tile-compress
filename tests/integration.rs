use std::collections::HashMap;

use lau_tile_compress::*;

// ---- Helper ----

fn make_tile(id: &str, content: Vec<f64>) -> RawTile {
    RawTile {
        id: id.to_string(),
        room_id: "room1".to_string(),
        timestamp: 1000,
        content,
        metadata: HashMap::new(),
    }
}

fn all_same(n: usize, val: f64) -> Vec<f64> {
    vec![val; n]
}

fn randomish(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7919 + 1) % 1000) as f64).collect()
}

// ===========================================================================
// Theorem 1: RLE — all-same data compresses to ~constant size
// ===========================================================================

#[test]
fn test_rle_all_same_compresses_constant() {
    let enc = RunLengthEncoder::new();
    let small = all_same(10, 5.0);
    let large = all_same(10_000, 5.0);
    let small_enc = enc.encode(&small);
    let large_enc = enc.encode(&large);
    assert_eq!(small_enc.len(), large_enc.len(), "RLE of all-same should be constant size");
}

#[test]
fn test_rle_all_same_roundtrip() {
    let enc = RunLengthEncoder::new();
    let data = all_same(100, 3.14);
    let encoded = enc.encode(&data);
    let decoded = enc.decode(&encoded);
    assert_eq!(data, decoded);
}

#[test]
fn test_rle_all_same_ratio_high() {
    let tile = make_tile("t1", all_same(1000, 1.0));
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress(&tile);
    let ratio = tile.byte_size() as f64 / compressed.data.len() as f64;
    assert!(ratio > 10.0, "RLE on all-same should have high ratio, got {ratio}");
}

// ===========================================================================
// Theorem 2: RLE — random data doesn't compress well
// ===========================================================================

#[test]
fn test_rle_random_ratio_near_one() {
    let tile = make_tile("t1", randomish(1000));
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress(&tile);
    let ratio = tile.byte_size() as f64 / compressed.data.len() as f64;
    assert!(ratio < 1.5, "Random data RLE ratio should be ~1.0, got {ratio}");
}

#[test]
fn test_rle_random_roundtrip() {
    let enc = RunLengthEncoder::new();
    let data = randomish(50);
    let encoded = enc.encode(&data);
    let decoded = enc.decode(&encoded);
    assert_eq!(data, decoded);
}

// ===========================================================================
// Theorem 3: Delta encoding — monotonic data has low entropy
// ===========================================================================

#[test]
fn test_delta_monotonic_low_entropy() {
    let enc = DeltaEncoder::new();
    let monotonic: Vec<f64> = (0..1000).map(|i| i as f64 * 2.0).collect();
    let entropy = enc.delta_entropy(&monotonic);
    let random_entropy = enc.delta_entropy(&randomish(1000));
    assert!(entropy < random_entropy, "Monotonic entropy ({entropy}) should be < random ({random_entropy})");
}

#[test]
fn test_delta_roundtrip() {
    let enc = DeltaEncoder::new();
    let data: Vec<f64> = vec![10.0, 15.0, 20.0, 25.0, 30.0];
    let encoded = enc.encode(&data);
    let decoded = enc.decode(encoded[0], &encoded[1..]);
    assert_eq!(data, decoded);
}

#[test]
fn test_delta_constant_data_zero_deltas() {
    let enc = DeltaEncoder::new();
    let data = all_same(10, 5.0);
    let deltas = enc.encode(&data);
    for &d in &deltas[1..] {
        assert_eq!(d, 0.0);
    }
}

#[test]
fn test_delta_entropy_empty() {
    let enc = DeltaEncoder::new();
    assert_eq!(enc.delta_entropy(&[]), 0.0);
    assert_eq!(enc.delta_entropy(&[1.0]), 0.0);
}

// ===========================================================================
// Theorem 4: Dictionary — repeated patterns get shorter codes
// ===========================================================================

#[test]
fn test_dictionary_repeated_patterns_compress() {
    let pattern: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let data: Vec<f64> = pattern.repeat(10);
    let mut enc = DictionaryEncoder::new();
    let _first = enc.encode(&data[..8]); // first 2 chunks — first is literal, second might be dict
    let _second = enc.encode(&data); // full 10 repetitions
    // After encoding, dictionary should have entries
    assert!(enc.dictionary_size() > 0);
}

#[test]
fn test_dictionary_second_occurrence_shorter() {
    let pattern: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let data: Vec<f64> = pattern.repeat(5);
    let mut enc = DictionaryEncoder::new();
    let encoded = enc.encode(&data);
    // First chunk is literal (33 bytes), subsequent are dict (3 bytes each)
    // Total should be much less than raw
    let raw_size = data.len() * 8;
    assert!(encoded.len() < raw_size, "Dict encoded {} should be < raw {}", encoded.len(), raw_size);
}

// ===========================================================================
// Theorem 5: Lossless roundtrip
// ===========================================================================

#[test]
fn test_rle_lossless_roundtrip() {
    let tile = make_tile("t1", vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    assert_eq!(tile.content, decompressed.content);
}

#[test]
fn test_delta_lossless_roundtrip() {
    let tile = make_tile("t1", vec![10.0, 20.0, 30.0, 40.0]);
    let comp = TileCompressor::new(CompressionStrategy::Delta, 1.0);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    assert_eq!(tile.content, decompressed.content);
}

#[test]
fn test_threshold_lossless_roundtrip_quality_1() {
    let tile = make_tile("t1", vec![1.0, 2.0, 3.0, 0.001]);
    let comp = TileCompressor::new(CompressionStrategy::Threshold, 1.0);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    assert_eq!(tile.content, decompressed.content);
    assert!(compressed.is_lossless());
}

#[test]
fn test_hybrid_lossless_roundtrip() {
    let tile = make_tile("t1", vec![5.0; 100]);
    let comp = TileCompressor::new(CompressionStrategy::Hybrid, 1.0);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    assert_eq!(tile.content, decompressed.content);
}

// ===========================================================================
// Theorem 6: Compression ratio is always ≥ 1.0 (can be worse for some strategies)
// ===========================================================================

#[test]
fn test_rle_ratio_not_below_half() {
    // RLE can't produce output larger than 12 bytes per element vs 8 raw
    let tile = make_tile("t1", randomish(100));
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress(&tile);
    // RLE might be slightly larger but should be reasonable
    assert!(compressed.data.len() < tile.byte_size() * 2);
}

// ===========================================================================
// Theorem 7: Semantic compression merges similar tiles
// ===========================================================================

#[test]
fn test_semantic_merges_similar() {
    let sc = SemanticCompressor::new(0.99);
    let t1 = make_tile("t1", vec![1.0, 0.0, 0.0]);
    let t2 = make_tile("t2", vec![1.0, 0.0, 0.0]); // identical
    let tiles = vec![t1, t2];
    let compressed = sc.compress(&tiles);
    assert_eq!(compressed.len(), 1, "Should merge identical tiles");
}

#[test]
fn test_semantic_keeps_dissimilar() {
    let sc = SemanticCompressor::new(0.99);
    let t1 = make_tile("t1", vec![1.0, 0.0, 0.0]);
    let t2 = make_tile("t2", vec![0.0, 1.0, 0.0]); // orthogonal
    let tiles = vec![t1, t2];
    let compressed = sc.compress(&tiles);
    assert_eq!(compressed.len(), 2, "Should keep dissimilar tiles separate");
}

#[test]
fn test_semantic_similarity_identical() {
    let sc = SemanticCompressor::new(0.9);
    let sim = sc.similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!((sim - 1.0).abs() < 1e-9);
}

#[test]
fn test_semantic_similarity_orthogonal() {
    let sc = SemanticCompressor::new(0.9);
    let sim = sc.similarity(&[1.0, 0.0], &[0.0, 1.0]);
    assert!(sim.abs() < 1e-9);
}

#[test]
fn test_semantic_merge_average() {
    let sc = SemanticCompressor::new(0.9);
    let t1 = make_tile("t1", vec![2.0, 4.0]);
    let t2 = make_tile("t2", vec![4.0, 8.0]);
    let merged = sc.merge(&[&t1, &t2]);
    assert_eq!(merged, vec![3.0, 6.0]);
}

#[test]
fn test_semantic_empty() {
    let sc = SemanticCompressor::new(0.9);
    let compressed = sc.compress(&[]);
    assert!(compressed.is_empty());
}

// ===========================================================================
// Theorem 8: Pipeline — sequential compression compounds ratios
// ===========================================================================

#[test]
fn test_pipeline_sequential() {
    let mut pipeline = CompressionPipeline::new();
    pipeline.add_stage(Box::new(CompressorStage::new(
        TileCompressor::new(CompressionStrategy::RunLength, 1.0),
    )));
    let tile = make_tile("t1", all_same(100, 5.0));
    let compressed = pipeline.compress(&tile);
    assert!(compressed.data.len() < tile.byte_size());
    assert!(pipeline.total_ratio() > 1.0);
}

#[test]
fn test_pipeline_two_stages() {
    let mut pipeline = CompressionPipeline::new();
    pipeline.add_stage(Box::new(CompressorStage::new(
        TileCompressor::new(CompressionStrategy::RunLength, 1.0),
    )));
    pipeline.add_stage(Box::new(CompressorStage::new(
        TileCompressor::new(CompressionStrategy::Threshold, 1.0),
    )));
    let tile = make_tile("t1", all_same(50, 7.0));
    let compressed = pipeline.compress(&tile);
    assert!(!compressed.data.is_empty());
}

// ===========================================================================
// Theorem 9: Quality=1.0 → lossless
// ===========================================================================

#[test]
fn test_quality_1_lossless() {
    let tile = make_tile("t1", vec![0.001, 0.002, 100.0]);
    let comp = TileCompressor::new(CompressionStrategy::Threshold, 1.0);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    assert_eq!(tile.content, decompressed.content);
    assert!(compressed.is_lossless());
}

// ===========================================================================
// Theorem 10: Quality=0.0 → maximum compression (most aggressive threshold)
// ===========================================================================

#[test]
fn test_quality_0_drops_small() {
    let tile = make_tile("t1", vec![0.001, 0.002, 100.0]);
    let comp = TileCompressor::new(CompressionStrategy::Threshold, 0.0);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    // At quality 0, threshold = (1-0)*max = max, so values < max are zeroed
    // 100.0 equals the max, so it survives; smaller values don't
    assert_eq!(decompressed.content[0], 0.0);
    assert_eq!(decompressed.content[1], 0.0);
    assert_eq!(decompressed.content[2], 100.0); // max value equals threshold
}

#[test]
fn test_quality_0_compresses_more() {
    let tile = make_tile("t1", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let comp_low = TileCompressor::new(CompressionStrategy::Threshold, 0.0);
    let comp_high = TileCompressor::new(CompressionStrategy::Threshold, 1.0);
    let c_low = comp_low.compress(&tile);
    let c_high = comp_high.compress(&tile);
    assert!(c_low.data.len() <= c_high.data.len());
}

// ===========================================================================
// Theorem 11: Batch compression stats are accurate
// ===========================================================================

#[test]
fn test_batch_stats_accurate() {
    let tiles: Vec<RawTile> = (0..10)
        .map(|i| make_tile(&format!("t{i}"), all_same(100, i as f64)))
        .collect();
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress_batch(&tiles);
    let stats = comp.stats(&tiles, &compressed);
    let manual_original: usize = tiles.iter().map(|t| t.byte_size()).sum();
    let manual_compressed: usize = compressed.iter().map(|c| c.data.len()).sum();
    assert_eq!(stats.original_bytes, manual_original);
    assert_eq!(stats.compressed_bytes, manual_compressed);
}

#[test]
fn test_batch_stats_ratio() {
    let tiles: Vec<RawTile> = (0..5)
        .map(|i| make_tile(&format!("t{i}"), all_same(500, i as f64)))
        .collect();
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress_batch(&tiles);
    let stats = comp.stats(&tiles, &compressed);
    assert!(stats.ratio() > 5.0, "Should compress well, got {}", stats.ratio());
    assert!(stats.savings_percent() > 80.0);
    assert!(stats.is_worthwhile());
}

// ===========================================================================
// Theorem 12: Empty tiles compress to near-zero
// ===========================================================================

#[test]
fn test_empty_tile_compresses_small() {
    let tile = make_tile("t1", vec![]);
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress(&tile);
    assert!(compressed.data.is_empty());
}

#[test]
fn test_empty_tile_stats() {
    let tile = make_tile("t1", vec![]);
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress(&tile);
    let stats = comp.stats(&[tile], &[compressed]);
    assert!(stats.ratio() >= 1.0);
}

// ===========================================================================
// Additional coverage tests
// ===========================================================================

#[test]
fn test_compression_stats_zero() {
    let stats = CompressionStats::new(0, 0);
    assert_eq!(stats.ratio(), 1.0);
    assert_eq!(stats.savings_percent(), 0.0);
    assert!(!stats.is_worthwhile());
}

#[test]
fn test_compression_stats_infinite() {
    let stats = CompressionStats::new(100, 0);
    assert!(stats.ratio().is_infinite());
    assert_eq!(stats.savings_percent(), 100.0);
}

#[test]
fn test_compressed_tile_is_lossless() {
    let ct = CompressedTile {
        id: "t1".into(),
        strategy: CompressionStrategy::RunLength,
        data: vec![],
        original_size: 100,
        quality: 1.0,
    };
    assert!(ct.is_lossless());
    let ct2 = CompressedTile {
        quality: 0.5,
        ..ct.clone()
    };
    assert!(!ct2.is_lossless());
}

#[test]
fn test_raw_tile_byte_size() {
    let mut tile = make_tile("id1", vec![1.0; 100]);
    tile.metadata.insert("key".into(), "value".into());
    let size = tile.byte_size();
    assert!(size > 0);
    // 100 f64s = 800 bytes minimum for content
    assert!(size >= 800);
}

#[test]
fn test_rle_empty() {
    let enc = RunLengthEncoder::new();
    assert!(enc.encode(&[]).is_empty());
    assert!(enc.decode(&[]).is_empty());
}

#[test]
fn test_delta_empty() {
    let enc = DeltaEncoder::new();
    let encoded = enc.encode(&[]);
    assert!(encoded.is_empty());
}

#[test]
fn test_delta_single() {
    let enc = DeltaEncoder::new();
    let data = vec![42.0];
    let encoded = enc.encode(&data);
    assert_eq!(encoded, vec![42.0]);
}

#[test]
fn test_dictionary_empty() {
    let mut enc = DictionaryEncoder::new();
    let encoded = enc.encode(&[]);
    assert!(encoded.is_empty());
}

#[test]
fn test_semantic_merge_empty() {
    let sc = SemanticCompressor::new(0.9);
    let merged = sc.merge(&[]);
    assert!(merged.is_empty());
}

#[test]
fn test_pipeline_empty_stages() {
    let mut pipeline = CompressionPipeline::new();
    let tile = make_tile("t1", vec![1.0, 2.0]);
    let compressed = pipeline.compress(&tile);
    assert!(!compressed.data.is_empty());
}

#[test]
fn test_compressor_stage_name() {
    let stage = CompressorStage::new(TileCompressor::new(CompressionStrategy::RunLength, 1.0));
    assert_eq!(stage.name(), "RunLength");
    let stage2 = CompressorStage::new(TileCompressor::new(CompressionStrategy::Delta, 1.0));
    assert_eq!(stage2.name(), "Delta");
    let stage3 = CompressorStage::new(TileCompressor::new(CompressionStrategy::Hybrid, 1.0));
    assert_eq!(stage3.name(), "Hybrid");
}

#[test]
fn test_dictionary_roundtrip_simple() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let mut enc = DictionaryEncoder::new();
    let encoded = enc.encode(&data);
    let decoded = enc.decode(&encoded);
    assert_eq!(data, decoded);
}

#[test]
fn test_dictionary_size_increases() {
    let mut enc = DictionaryEncoder::new();
    assert_eq!(enc.dictionary_size(), 0);
    enc.encode(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(enc.dictionary_size(), 1);
}

#[test]
fn test_similarity_different_lengths() {
    let sc = SemanticCompressor::new(0.9);
    assert_eq!(sc.similarity(&[1.0], &[1.0, 2.0]), 0.0);
}

#[test]
fn test_similarity_empty() {
    let sc = SemanticCompressor::new(0.9);
    assert_eq!(sc.similarity(&[], &[]), 0.0);
}

#[test]
fn test_similarity_zero_vector() {
    let sc = SemanticCompressor::new(0.9);
    assert_eq!(sc.similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
}

#[test]
fn test_dictionary_default() {
    let enc = DictionaryEncoder::default();
    assert_eq!(enc.dictionary_size(), 0);
}

#[test]
fn test_run_length_encoder_default() {
    let enc = RunLengthEncoder::default();
    let data = vec![1.0, 1.0, 2.0];
    let encoded = enc.encode(&data);
    let decoded = enc.decode(&encoded);
    assert_eq!(data, decoded);
}

#[test]
fn test_delta_default() {
    let enc = DeltaEncoder::default();
    let data = vec![1.0, 2.0, 3.0];
    let encoded = enc.encode(&data);
    assert_eq!(encoded.len(), 3);
}

#[test]
fn test_pipeline_default() {
    let pipeline = CompressionPipeline::default();
    assert!(pipeline.stages.is_empty());
    assert_eq!(pipeline.total_ratio(), 1.0);
}

#[test]
fn test_serde_roundtrip_compressed_tile() {
    let ct = CompressedTile {
        id: "t1".into(),
        strategy: CompressionStrategy::RunLength,
        data: vec![1, 2, 3],
        original_size: 100,
        quality: 0.8,
    };
    let json = serde_json::to_string(&ct).unwrap();
    let back: CompressedTile = serde_json::from_str(&json).unwrap();
    assert_eq!(ct.id, back.id);
    assert_eq!(ct.strategy, back.strategy);
    assert_eq!(ct.data, back.data);
}

#[test]
fn test_serde_roundtrip_raw_tile() {
    let tile = make_tile("t1", vec![1.0, 2.0]);
    let json = serde_json::to_string(&tile).unwrap();
    let back: RawTile = serde_json::from_str(&json).unwrap();
    assert_eq!(tile.id, back.id);
    assert_eq!(tile.content, back.content);
}

#[test]
fn test_serde_roundtrip_compression_strategy() {
    let strategies = vec![
        CompressionStrategy::RunLength,
        CompressionStrategy::Dictionary,
        CompressionStrategy::Delta,
        CompressionStrategy::Threshold,
        CompressionStrategy::Semantic,
        CompressionStrategy::Hybrid,
    ];
    for s in &strategies {
        let json = serde_json::to_string(s).unwrap();
        let back: CompressionStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(*s, back);
    }
}

#[test]
fn test_compression_stats_serde() {
    let stats = CompressionStats::new(1000, 500);
    let json = serde_json::to_string(&stats).unwrap();
    let back: CompressionStats = serde_json::from_str(&json).unwrap();
    assert_eq!(stats.original_bytes, back.original_bytes);
    assert_eq!(stats.compressed_bytes, back.compressed_bytes);
}

#[test]
fn test_batch_compress_count_matches() {
    let tiles: Vec<RawTile> = (0..7)
        .map(|i| make_tile(&format!("t{i}"), vec![i as f64; 10]))
        .collect();
    let comp = TileCompressor::new(CompressionStrategy::RunLength, 1.0);
    let compressed = comp.compress_batch(&tiles);
    assert_eq!(compressed.len(), tiles.len());
}

#[test]
fn test_threshold_quality_partial() {
    let tile = make_tile("t1", vec![0.1, 50.0, 0.2, 100.0]);
    let comp = TileCompressor::new(CompressionStrategy::Threshold, 0.5);
    let compressed = comp.compress(&tile);
    let decompressed = comp.decompress(&compressed);
    // With quality 0.5, threshold = 0.5 * 100 = 50, so values < 50 are dropped
    assert_eq!(decompressed.content[0], 0.0);
    assert_eq!(decompressed.content[2], 0.0);
}
