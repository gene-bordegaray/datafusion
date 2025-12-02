// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::prelude::{col, ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_expr::SortExpr;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Configuration for benchmark data size
#[derive(Debug, Clone, Copy)]
struct DataConfig {
    num_partitions: usize,
    rows_per_partition: usize,
    target_partitions: usize,
}

impl DataConfig {
    fn small() -> Self {
        Self {
            num_partitions: 5,
            rows_per_partition: 200_000,
            target_partitions: 3,
        }
    }

    fn medium() -> Self {
        Self {
            num_partitions: 20,
            rows_per_partition: 1_250_000,
            target_partitions: 10,
        }
    }

    fn large() -> Self {
        Self {
            num_partitions: 40,
            rows_per_partition: 10_000_000,
            target_partitions: 15,
        }
    }

    /// Get target benchmark time in seconds based on size
    fn target_time_secs(&self) -> u64 {
        if self.num_partitions == Self::small().num_partitions {
            10
        } else if self.num_partitions == Self::medium().num_partitions {
            30
        } else {
            90
        }
    }

    /// Get configuration based on environment variable BENCH_SIZE
    /// Defaults to Small
    fn from_env() -> Self {
        match std::env::var("BENCH_SIZE").as_deref() {
            Ok("small") | Ok("SMALL") => Self::small(),
            Ok("medium") | Ok("MEDIUM") => Self::medium(),
            Ok("large") | Ok("LARGE") => Self::large(),
            _ => {
                println!(
                    "Using SMALL dataset (set BENCH_SIZE=small|medium|large to change)"
                );
                Self::small()
            }
        }
    }
}

/// Generate partitioned data for benchmarking
fn generate_partitioned_data(
    base_dir: &Path,
    num_partitions: usize,
    rows_per_partition: usize,
) {
    // Wider schema with 14 columns (category is partition column, not in Parquet files)
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("col1", DataType::Float64, false),
        Field::new("col2", DataType::Float64, false),
        Field::new("col3", DataType::Int64, false),
        Field::new("col4", DataType::Utf8, false),
        Field::new("col5", DataType::Int32, false),
        Field::new("col6", DataType::Float64, false),
        Field::new("col7", DataType::Int64, false),
        Field::new("col8", DataType::Utf8, false),
        Field::new("col9", DataType::Int32, false),
        Field::new("col10", DataType::Float64, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();

    for i in 0..num_partitions {
        let category = format!("cat_{}", i);
        let part_dir = base_dir.join(format!("category={}", category));
        fs::create_dir_all(&part_dir).unwrap();
        let file_path = part_dir.join("data.parquet");
        let file = File::create(file_path).unwrap();

        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), Some(props.clone())).unwrap();

        let ids: Vec<i64> = (0..rows_per_partition)
            .map(|x| (i * rows_per_partition + x) as i64)
            .collect();
        let amounts: Vec<f64> = (0..rows_per_partition)
            .map(|x| ((x % 1000) as f64) + ((x % 100) as f64 * 0.1))
            .collect();
        let quantities: Vec<i32> =
            (0..rows_per_partition).map(|x| (x % 100) as i32).collect();
        let timestamps: Vec<i64> = (0..rows_per_partition)
            .map(|x| 1700000000000 + (x as i64 * 1000)) // 1 second apart
            .collect();

        // Additional dummy columns
        let col1: Vec<f64> = (0..rows_per_partition).map(|x| (x as f64) * 1.5).collect();
        let col2: Vec<f64> = (0..rows_per_partition).map(|x| (x as f64) * 2.3).collect();
        let col3: Vec<i64> = (0..rows_per_partition).map(|x| (x % 500) as i64).collect();
        let col4: Vec<String> = (0..rows_per_partition)
            .map(|x| format!("val_{:04x}", x % 200))
            .collect();
        let col5: Vec<i32> = (0..rows_per_partition).map(|x| (x % 50) as i32).collect();
        let col6: Vec<f64> = (0..rows_per_partition).map(|x| (x as f64) / 10.0).collect();
        let col7: Vec<i64> = (0..rows_per_partition).map(|x| (x * 7) as i64).collect();
        let col8: Vec<String> = (0..rows_per_partition)
            .map(|x| format!("item_{}", x % 300))
            .collect();
        let col9: Vec<i32> = (0..rows_per_partition).map(|x| (x % 25) as i32).collect();
        let col10: Vec<f64> =
            (0..rows_per_partition).map(|x| (x as f64) * 0.75).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(Float64Array::from(amounts)),
                Arc::new(Int32Array::from(quantities)),
                Arc::new(TimestampMillisecondArray::from(timestamps)),
                Arc::new(Float64Array::from(col1)),
                Arc::new(Float64Array::from(col2)),
                Arc::new(Int64Array::from(col3)),
                Arc::new(StringArray::from(col4)),
                Arc::new(Int32Array::from(col5)),
                Arc::new(Float64Array::from(col6)),
                Arc::new(Int64Array::from(col7)),
                Arc::new(StringArray::from(col8)),
                Arc::new(Int32Array::from(col9)),
                Arc::new(Float64Array::from(col10)),
            ],
        )
        .unwrap();

        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}

/// Generate sorted partitioned data for benchmarking sort-preserving optimizations
/// Data is sorted by timestamp within each partition, enabling SortPreservingMerge
fn generate_sorted_partitioned_data(
    base_dir: &Path,
    num_partitions: usize,
    rows_per_partition: usize,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("col1", DataType::Float64, false),
        Field::new("col2", DataType::Float64, false),
        Field::new("col3", DataType::Int64, false),
        Field::new("col4", DataType::Utf8, false),
        Field::new("col5", DataType::Int32, false),
        Field::new("col6", DataType::Float64, false),
        Field::new("col7", DataType::Int64, false),
        Field::new("col8", DataType::Utf8, false),
        Field::new("col9", DataType::Int32, false),
        Field::new("col10", DataType::Float64, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();

    for i in 0..num_partitions {
        let category = format!("cat_{}", i);
        let part_dir = base_dir.join(format!("category={}", category));
        fs::create_dir_all(&part_dir).unwrap();
        let file_path = part_dir.join("data.parquet");
        let file = File::create(file_path).unwrap();

        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), Some(props.clone())).unwrap();

        // Generate data that's already sorted by timestamp
        let ids: Vec<i64> = (0..rows_per_partition)
            .map(|x| (i * rows_per_partition + x) as i64)
            .collect();
        let amounts: Vec<f64> = (0..rows_per_partition)
            .map(|x| ((x % 1000) as f64) + ((x % 100) as f64 * 0.1))
            .collect();
        let quantities: Vec<i32> =
            (0..rows_per_partition).map(|x| (x % 100) as i32).collect();
        let timestamps: Vec<i64> = (0..rows_per_partition)
            .map(|x| 1700000000000 + (x as i64 * 1000))
            .collect();

        // Additional dummy columns
        let col1: Vec<f64> = (0..rows_per_partition).map(|x| (x as f64) * 1.5).collect();
        let col2: Vec<f64> = (0..rows_per_partition).map(|x| (x as f64) * 2.3).collect();
        let col3: Vec<i64> = (0..rows_per_partition).map(|x| (x % 500) as i64).collect();
        let col4: Vec<String> = (0..rows_per_partition)
            .map(|x| format!("val_{:04x}", x % 200))
            .collect();
        let col5: Vec<i32> = (0..rows_per_partition).map(|x| (x % 50) as i32).collect();
        let col6: Vec<f64> = (0..rows_per_partition).map(|x| (x as f64) / 10.0).collect();
        let col7: Vec<i64> = (0..rows_per_partition).map(|x| (x * 7) as i64).collect();
        let col8: Vec<String> = (0..rows_per_partition)
            .map(|x| format!("item_{}", x % 300))
            .collect();
        let col9: Vec<i32> = (0..rows_per_partition).map(|x| (x % 25) as i32).collect();
        let col10: Vec<f64> =
            (0..rows_per_partition).map(|x| (x as f64) * 0.75).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(Float64Array::from(amounts)),
                Arc::new(Int32Array::from(quantities)),
                Arc::new(TimestampMillisecondArray::from(timestamps)),
                Arc::new(Float64Array::from(col1)),
                Arc::new(Float64Array::from(col2)),
                Arc::new(Int64Array::from(col3)),
                Arc::new(StringArray::from(col4)),
                Arc::new(Int32Array::from(col5)),
                Arc::new(Float64Array::from(col6)),
                Arc::new(Int64Array::from(col7)),
                Arc::new(StringArray::from(col8)),
                Arc::new(Int32Array::from(col9)),
                Arc::new(Float64Array::from(col10)),
            ],
        )
        .unwrap();

        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}

/// Save execution plans to file
async fn save_plans(
    table_name: &str,
    table_path: &str,
    output_file: &Path,
    target_partitions: usize,
    query: &str,
    file_sort_order: Option<Vec<Vec<SortExpr>>>,
) {
    let mut file = File::create(output_file).unwrap();

    writeln!(file, "KeyPartitioned Aggregation Benchmark Plans\n").unwrap();
    writeln!(file, "Query: {query}\n").unwrap();

    // Optimized plan
    let config_opt = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .set_bool(
            "datafusion.execution.listing_table_preserve_partition_values",
            true,
        );
    let ctx_opt = SessionContext::new_with_config(config_opt);
    let mut options = ParquetReadOptions {
        table_partition_cols: vec![("category".to_string(), DataType::Utf8)],
        ..Default::default()
    };
    if let Some(sort_order) = file_sort_order.clone() {
        options.file_sort_order = sort_order;
    }
    ctx_opt
        .register_parquet(table_name, table_path, options.clone())
        .await
        .unwrap();

    let df_opt = ctx_opt.sql(query).await.unwrap();
    let plan_opt = df_opt
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    writeln!(file, "=== WITH KeyPartitioned Optimization ===").unwrap();
    writeln!(file, "{}\n", pretty_format_batches(&plan_opt).unwrap()).unwrap();

    // Unoptimized plan
    let config_unopt = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .set_bool(
            "datafusion.execution.listing_table_preserve_partition_values",
            false,
        );
    let ctx_unopt = SessionContext::new_with_config(config_unopt);
    let mut options = ParquetReadOptions {
        table_partition_cols: vec![("category".to_string(), DataType::Utf8)],
        ..Default::default()
    };
    if let Some(sort_order) = file_sort_order {
        options.file_sort_order = sort_order;
    }
    ctx_unopt
        .register_parquet(table_name, table_path, options)
        .await
        .unwrap();

    let df_unopt = ctx_unopt.sql(query).await.unwrap();
    let plan_unopt = df_unopt
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    writeln!(file, "=== WITHOUT KeyPartitioned Optimization ===").unwrap();
    writeln!(file, "{}", pretty_format_batches(&plan_unopt).unwrap()).unwrap();
}

/// Helper function to run a benchmark configuration
fn benchmark_config(
    c: &mut Criterion,
    name: &str,
    table_name: &str,
    partitions: usize,
    rows_per_partition: usize,
    target_partitions: usize,
    query: &str,
    file_sort_order: Option<Vec<Vec<SortExpr>>>,
) {
    let tmp_dir = TempDir::new().unwrap();
    if file_sort_order.is_some() {
        generate_sorted_partitioned_data(tmp_dir.path(), partitions, rows_per_partition);
    } else {
        generate_partitioned_data(tmp_dir.path(), partitions, rows_per_partition);
    }
    let table_path = tmp_dir.path().to_str().unwrap();
    let rt = Runtime::new().unwrap();

    // Save execution plans if SAVE_PLANS env var is set
    if std::env::var("SAVE_PLANS").is_ok() {
        let output_filename = format!("{}_plans.txt", name);
        let output_path = Path::new(&output_filename);
        rt.block_on(save_plans(
            table_name,
            table_path,
            output_path,
            target_partitions,
            query,
            file_sort_order.clone(),
        ));
        println!("Execution plans saved to {}", output_path.display());
    }

    let mut group = c.benchmark_group(name);

    let sort_order_clone1 = file_sort_order.clone();
    group.bench_function("with_key_partitioned", |b| {
        b.to_async(&rt).iter(|| async {
            let config = SessionConfig::new()
                .with_target_partitions(target_partitions)
                .set_bool(
                    "datafusion.execution.listing_table_preserve_partition_values",
                    true,
                );
            let ctx = SessionContext::new_with_config(config);

            let mut options = ParquetReadOptions {
                table_partition_cols: vec![("category".to_string(), DataType::Utf8)],
                ..Default::default()
            };
            if let Some(ref sort_order) = sort_order_clone1 {
                options.file_sort_order = sort_order.clone();
            }

            ctx.register_parquet(table_name, table_path, options)
                .await
                .unwrap();

            let df = ctx.sql(query).await.unwrap();
            df.collect().await.unwrap()
        })
    });

    let sort_order_clone2 = file_sort_order;
    group.bench_function("without_key_partitioned", |b| {
        b.to_async(&rt).iter(|| async {
            let config = SessionConfig::new()
                .with_target_partitions(target_partitions)
                .set_bool(
                    "datafusion.execution.listing_table_preserve_partition_values",
                    false,
                );
            let ctx = SessionContext::new_with_config(config);

            let mut options = ParquetReadOptions {
                table_partition_cols: vec![("category".to_string(), DataType::Utf8)],
                ..Default::default()
            };
            if let Some(ref sort_order) = sort_order_clone2 {
                options.file_sort_order = sort_order.clone();
            }

            ctx.register_parquet(table_name, table_path, options)
                .await
                .unwrap();

            let df = ctx.sql(query).await.unwrap();
            df.collect().await.unwrap()
        })
    });

    group.finish();
}

/// Simple Aggregation Benchmark
///
/// This benchmark uses **less partitions** (fewer, larger files) to demonstrate
/// the KeyPartitioned optimization eliminating hash repartition. Using fewer
/// partitions limits the I/O cost of reading files, emphasizing the
/// performance enhancement of eliminating the repartition elimination.
///
/// **Performance expectations**:
/// - With optimization: ~1.1-1.3x speedup
/// - Benefit: Eliminates hash repartition overhead
///
/// **What's being optimized**:
/// - Eliminates: RepartitionExec (hash shuffle)
/// - Changes: Partial+FinalPartitioned → SinglePartitioned aggregation
fn simple_aggregation_bench(c: &mut Criterion, config: DataConfig) {
    let query = "SELECT category, COUNT(*), SUM(amount), AVG(amount) \
                 FROM facts \
                 WHERE timestamp > TIMESTAMP '2023-11-15 00:00:00' \
                 GROUP BY category";

    benchmark_config(
        c,
        "simple_aggregation",
        "facts",
        config.num_partitions,
        config.rows_per_partition,
        config.target_partitions,
        query,
        None, // no sort order
    );
}

/// Complex Aggregation Benchmark
///
/// This benchmark uses **less partitions** (fewer, larger files) to demonstrate
/// the KeyPartitioned optimization with expensive aggregations (STDDEV). The
/// reason for fewer partitions is due to the same reasoning as the simple
/// aggregation query. Now, the aggregation takes up and even more significant
/// amount of total processing time, making the key partitioning speedups more
/// prevelant.
///
/// **Performance expectations**:
/// - With optimization: ~1.2-1.4x speedup
/// - Benefit: Eliminates hash repartition + reduces expensive merge operations
/// - Why fewer partitions: Reduces I/O overhead, makes merge cost visible
///
/// **What's being optimized**:
/// - Eliminates: RepartitionExec ()
/// - Eliminates: Expensive STDDEV state merging across partitions
/// - Changes: Partial+FinalPartitioned → SinglePartitioned aggregation
fn complex_aggregation_bench(c: &mut Criterion, config: DataConfig) {
    let query = "SELECT category, \
                        COUNT(*), \
                        AVG(amount), \
                        STDDEV(amount), \
                        MIN(amount), \
                        MAX(amount), \
                        SUM(quantity), \
                        AVG(quantity) \
                 FROM facts \
                 WHERE timestamp > TIMESTAMP '2023-11-15 00:00:00' \
                 GROUP BY category";

    benchmark_config(
        c,
        "complex_aggregation",
        "facts",
        config.num_partitions,
        config.rows_per_partition,
        config.target_partitions,
        query,
        None, // no sort order
    );
}

/// Sorted Aggregation Benchmark
///
/// This benchmark uses **more partitioning with more data** to demonstrate
/// the combined benefit of KeyPartitioned optimization and sort preservation.
///
/// **Data volume**: Uses 1.5-2× more data than base config to show sort elimination benefit.
///
/// **Performance expectations**:
/// - With optimization: ~2.5-4.0x speedup
/// - Benefit: Eliminates hash repartition + eliminates expensive sort operations
///
/// **What's being optimized**:
/// - Eliminates: RepartitionExec (hash shuffle)
/// - Eliminates: SortExec (sorting large partitions twice is VERY expensive!)
/// - Uses: Sort-preserving aggregation (ordering_mode=Sorted)
/// - Result: Only SortPreservingMergeExec remains (very cheap)
fn sorted_aggregation_bench(c: &mut Criterion, config: DataConfig) {
    let sorted_config = DataConfig {
        num_partitions: match config.num_partitions {
            5 => 25,
            20 => 100,
            _ => 150,
        },
        rows_per_partition: match config.num_partitions {
            5 => 40_000,
            20 => 500_000,
            _ => 2_000_000,
        },
        target_partitions: match config.num_partitions {
            5 => 12,
            20 => 30,
            _ => 40,
        },
    };

    println!(
        "Sorted aggregation using: {} partitions × {} rows = {} total rows",
        sorted_config.num_partitions,
        sorted_config.rows_per_partition,
        sorted_config.num_partitions * sorted_config.rows_per_partition
    );
    println!("  (Using more data than base to amplify sort elimination benefit)\n");

    let query = "SELECT category, \
                        COUNT(*) as cnt, \
                        SUM(amount) as total_amount, \
                        AVG(amount) as avg_amount, \
                        MAX(amount) as max_amount \
                 FROM facts \
                 WHERE timestamp > TIMESTAMP '2023-11-15 00:00:00' \
                 GROUP BY category \
                 ORDER BY category";

    // Specify that files are sorted by category ascending
    let file_sort_order = vec![vec![col("category").sort(true, false)]];

    benchmark_config(
        c,
        "sorted_aggregation",
        "facts",
        sorted_config.num_partitions,
        sorted_config.rows_per_partition,
        sorted_config.target_partitions,
        query,
        Some(file_sort_order),
    );
}

fn run_benchmark(c: &mut Criterion) {
    let config = DataConfig::from_env();

    println!("\n=== KeyPartitioned Aggregation Benchmarks ===");
    println!(
        "Base configuration: {} partitions × {} rows = {} total rows",
        config.num_partitions,
        config.rows_per_partition,
        config.num_partitions * config.rows_per_partition
    );
    println!("Target time: {}s", config.target_time_secs());
    println!("\nSimple/Complex benchmarks use base configuration.");
    println!("Sorted benchmark uses more partitions + more data for dramatic results.\n");

    simple_aggregation_bench(c, config);
    complex_aggregation_bench(c, config);
    sorted_aggregation_bench(c, config);
}

criterion_group! {
    name = benches;
    config = {
        let config = DataConfig::from_env();
        Criterion::default()
            .measurement_time(std::time::Duration::from_secs(config.target_time_secs()))
    };
    targets = run_benchmark
}
criterion_main!(benches);
