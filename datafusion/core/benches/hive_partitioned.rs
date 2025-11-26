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

use arrow::array::{ArrayRef, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Generate partitioned data for benchmarking
/// Creates `num_partitions` directories, each with one parquet file containing `rows_per_partition`.
fn generate_partitioned_data(
    base_dir: &Path,
    num_partitions: usize,
    rows_per_partition: usize,
) {
    let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int32, true)]));

    let props = WriterProperties::builder().build();

    for i in 0..num_partitions {
        let part_dir = base_dir.join(format!("part_col={i}"));
        fs::create_dir_all(&part_dir).unwrap();
        let file_path = part_dir.join("data.parquet");
        let file = File::create(file_path).unwrap();

        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), Some(props.clone())).unwrap();

        // Generate data: just a simple sequence
        let vals: Vec<i32> = (0..rows_per_partition).map(|x| x as i32).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vals)) as ArrayRef],
        )
        .unwrap();

        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}

/// Generate dimension table for join benchmarks
fn generate_dimension_table(base_dir: &Path, num_dimensions: usize) {
    use arrow::array::StringArray;

    let schema = Arc::new(Schema::new(vec![
        Field::new("dim_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let props = WriterProperties::builder().build();
    let dim_dir = base_dir.join("dim");
    fs::create_dir_all(&dim_dir).unwrap();
    let file_path = dim_dir.join("dimension.parquet");
    let file = File::create(file_path).unwrap();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();

    // Generate dimension data: dim_id from 0 to num_dimensions-1
    let dim_ids: Vec<i32> = (0..num_dimensions as i32).collect();
    let categories: Vec<String> = (0..num_dimensions)
        .map(|i| format!("Category_{}", i % 10))  // 10 categories cycling
        .collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(dim_ids)) as ArrayRef,
            Arc::new(StringArray::from(categories)) as ArrayRef,
        ],
    )
    .unwrap();

    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Save execution plans to file
async fn save_plans(fact_table_path: &str, dim_table_path: &str, output_file: &Path) {
    let agg_query = "SELECT part_col, count(*), sum(val), avg(val) FROM fact GROUP BY part_col";
    let join_query = "SELECT f.part_col, COUNT(*), MAX(d.category) FROM fact f JOIN dim d ON f.part_col = d.dim_id GROUP BY f.part_col";
    let mut file = File::create(output_file).unwrap();

    writeln!(file, "KeyPartitioned Optimization Benchmark Plans\n").unwrap();

    let fact_options = ParquetReadOptions {
        table_partition_cols: vec![("part_col".to_string(), DataType::Int32)],
        ..Default::default()
    };

    // Test 1: Simple aggregation on partition key
    writeln!(file, "========================================").unwrap();
    writeln!(file, "TEST 1: Simple Aggregation on Partition Key").unwrap();
    writeln!(file, "Query: {}\n", agg_query).unwrap();

    // Optimized
    let config_opt = SessionConfig::new().with_target_partitions(20).set_bool(
        "datafusion.execution.listing_table_preserve_partition_values",
        true,
    );
    let ctx_opt = SessionContext::new_with_config(config_opt);
    ctx_opt
        .register_parquet("fact", fact_table_path, fact_options.clone())
        .await
        .unwrap();

    let df_opt = ctx_opt.sql(agg_query).await.unwrap();
    let plan_opt = df_opt
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    writeln!(file, "=== WITH KeyPartitioned ===").unwrap();
    writeln!(file, "{}\n", pretty_format_batches(&plan_opt).unwrap()).unwrap();

    // Unoptimized
    let config_unopt = SessionConfig::new().with_target_partitions(20).set_bool(
        "datafusion.execution.listing_table_preserve_partition_values",
        false,
    );
    let ctx_unopt = SessionContext::new_with_config(config_unopt);
    ctx_unopt
        .register_parquet("fact", fact_table_path, fact_options.clone())
        .await
        .unwrap();

    let df_unopt = ctx_unopt.sql(agg_query).await.unwrap();
    let plan_unopt = df_unopt
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    writeln!(file, "=== WITHOUT KeyPartitioned ===").unwrap();
    writeln!(file, "{}\n", pretty_format_batches(&plan_unopt).unwrap()).unwrap();

    // Test 2: Join + aggregation
    writeln!(file, "========================================").unwrap();
    writeln!(file, "TEST 2: Join + Aggregation on Partition Key").unwrap();
    writeln!(file, "Query: {}\n", join_query).unwrap();

    // Optimized with join
    let config_opt_join = SessionConfig::new().with_target_partitions(20).set_bool(
        "datafusion.execution.listing_table_preserve_partition_values",
        true,
    );
    let ctx_opt_join = SessionContext::new_with_config(config_opt_join);
    ctx_opt_join
        .register_parquet("fact", fact_table_path, fact_options.clone())
        .await
        .unwrap();
    ctx_opt_join
        .register_parquet("dim", dim_table_path, ParquetReadOptions::default())
        .await
        .unwrap();

    let df_opt_join = ctx_opt_join.sql(join_query).await.unwrap();
    let plan_opt_join = df_opt_join
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    writeln!(file, "=== WITH KeyPartitioned (Join Propagation) ===").unwrap();
    writeln!(file, "{}\n", pretty_format_batches(&plan_opt_join).unwrap()).unwrap();

    // Unoptimized with join
    let config_unopt_join = SessionConfig::new().with_target_partitions(20).set_bool(
        "datafusion.execution.listing_table_preserve_partition_values",
        false,
    );
    let ctx_unopt_join = SessionContext::new_with_config(config_unopt_join);
    ctx_unopt_join
        .register_parquet("fact", fact_table_path, fact_options)
        .await
        .unwrap();
    ctx_unopt_join
        .register_parquet("dim", dim_table_path, ParquetReadOptions::default())
        .await
        .unwrap();

    let df_unopt_join = ctx_unopt_join.sql(join_query).await.unwrap();
    let plan_unopt_join = df_unopt_join
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    writeln!(file, "=== WITHOUT KeyPartitioned ===").unwrap();
    writeln!(file, "{}", pretty_format_batches(&plan_unopt_join).unwrap()).unwrap();
}

fn run_benchmark(c: &mut Criterion) {
    // Benchmark KeyPartitioned optimization for aggregations on Hive-partitioned tables
    // 20 partitions × 500K rows = 10M total rows
    let partitions = 20;
    let rows_per_partition = 500_000;
    let tmp_dir = TempDir::new().unwrap();

    generate_partitioned_data(tmp_dir.path(), partitions, rows_per_partition);
    generate_dimension_table(tmp_dir.path(), partitions);  // Generate dimension table matching partition count

    let fact_table_path = tmp_dir.path().to_str().unwrap();
    let dim_table_path = tmp_dir.path().join("dim");
    let dim_table_str = dim_table_path.to_str().unwrap();

    let rt = Runtime::new().unwrap();

    // Save execution plans if SAVE_PLANS env var is set
    if std::env::var("SAVE_PLANS").is_ok() {
        let output_path = Path::new("hive_partitioned_plans.txt");
        rt.block_on(save_plans(fact_table_path, dim_table_str, output_path));
        println!("Execution plans saved to {}", output_path.display());
    }

    let query = "SELECT part_col, count(*), sum(val), avg(val) FROM t GROUP BY part_col";
    let mut group = c.benchmark_group("hive_partitioned_agg");

    group.bench_function("with_key_partitioned", |b| {
        b.to_async(&rt).iter(|| async {
            let config = SessionConfig::new().with_target_partitions(20).set_bool(
                "datafusion.execution.listing_table_preserve_partition_values",
                true,
            );
            let ctx = SessionContext::new_with_config(config);

            let options = ParquetReadOptions {
                table_partition_cols: vec![("part_col".to_string(), DataType::Int32)],
                ..Default::default()
            };

            ctx.register_parquet("t", fact_table_path, options)
                .await
                .unwrap();

            let df = ctx.sql(query).await.unwrap();
            df.collect().await.unwrap();
        })
    });

    group.bench_function("without_key_partitioned", |b| {
        b.to_async(&rt).iter(|| async {
            let config = SessionConfig::new().with_target_partitions(20).set_bool(
                "datafusion.execution.listing_table_preserve_partition_values",
                false,
            );
            let ctx = SessionContext::new_with_config(config);

            let options = ParquetReadOptions {
                table_partition_cols: vec![("part_col".to_string(), DataType::Int32)],
                ..Default::default()
            };

            ctx.register_parquet("t", fact_table_path, options)
                .await
                .unwrap();

            let df = ctx.sql(query).await.unwrap();
            df.collect().await.unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, run_benchmark);
criterion_main!(benches);
