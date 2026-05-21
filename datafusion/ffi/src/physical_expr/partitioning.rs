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

use std::sync::Arc;

use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_physical_expr::{
    LexOrdering, Partitioning, PhysicalSortExpr, RangePartitioning, SplitPoint,
};
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use log::error;
use prost::Message;
use stabby::vec::Vec as SVec;

use crate::physical_expr::FFI_PhysicalExpr;
use crate::physical_expr::sort::FFI_PhysicalSortExpr;

/// A stable struct for sharing [`SplitPoint`] across FFI boundaries.
///
/// See [`SplitPoint`] for full documentation.
#[repr(C)]
#[derive(Debug)]
pub struct FFI_RangeSplitPoint {
    values: SVec<SVec<u8>>,
}

/// A stable struct for sharing [`Partitioning`] across FFI boundaries.
/// See ['Partitioning'] for the meaning of each variant.
#[repr(C)]
#[derive(Debug)]
pub enum FFI_Partitioning {
    RoundRobinBatch(usize),
    Hash(SVec<FFI_PhysicalExpr>, usize),
    UnknownPartitioning(usize),
    Range(SVec<FFI_PhysicalSortExpr>, SVec<FFI_RangeSplitPoint>),
}

impl From<&Partitioning> for FFI_Partitioning {
    fn from(value: &Partitioning) -> Self {
        match value {
            Partitioning::RoundRobinBatch(size) => Self::RoundRobinBatch(*size),
            Partitioning::Hash(exprs, size) => {
                let exprs = exprs
                    .iter()
                    .map(Arc::clone)
                    .map(FFI_PhysicalExpr::from)
                    .collect();
                Self::Hash(exprs, *size)
            }
            Partitioning::Range(range) => range_to_ffi(range).unwrap_or_else(|err| {
                error!(
                    "Unable to convert range partitioning to FFI. Falling back to UnknownPartitioning. {err}"
                );
                Self::UnknownPartitioning(range.partition_count())
            }),
            Partitioning::UnknownPartitioning(size) => Self::UnknownPartitioning(*size),
        }
    }
}

impl From<&FFI_Partitioning> for Partitioning {
    fn from(value: &FFI_Partitioning) -> Self {
        value.try_to_partitioning().unwrap_or_else(|err| {
            error!(
                "Unable to convert FFI partitioning to DataFusion. Falling back to UnknownPartitioning. {err}"
            );
            value.fallback_partitioning()
        })
    }
}

impl FFI_Partitioning {
    pub(crate) fn try_to_partitioning(&self) -> Result<Partitioning, DataFusionError> {
        match self {
            FFI_Partitioning::RoundRobinBatch(size) => {
                Ok(Partitioning::RoundRobinBatch(*size))
            }
            FFI_Partitioning::Hash(exprs, size) => {
                let exprs = exprs.iter().map(<Arc<dyn PhysicalExpr>>::from).collect();
                Ok(Partitioning::Hash(exprs, *size))
            }
            FFI_Partitioning::UnknownPartitioning(size) => {
                Ok(Partitioning::UnknownPartitioning(*size))
            }
            FFI_Partitioning::Range(ordering, split_points) => {
                ffi_range_to_partitioning(ordering, split_points)
            }
        }
    }

    fn fallback_partitioning(&self) -> Partitioning {
        match self {
            FFI_Partitioning::Range(_, split_points) => {
                Partitioning::UnknownPartitioning(split_points.len() + 1)
            }
            FFI_Partitioning::RoundRobinBatch(size)
            | FFI_Partitioning::UnknownPartitioning(size) => {
                Partitioning::UnknownPartitioning(*size)
            }
            FFI_Partitioning::Hash(_, size) => Partitioning::UnknownPartitioning(*size),
        }
    }
}

impl TryFrom<&SplitPoint> for FFI_RangeSplitPoint {
    type Error = DataFusionError;

    fn try_from(value: &SplitPoint) -> Result<Self, Self::Error> {
        let values = value
            .values()
            .iter()
            .map(encode_scalar_value)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect();

        Ok(Self { values })
    }
}

impl TryFrom<&FFI_RangeSplitPoint> for SplitPoint {
    type Error = DataFusionError;

    fn try_from(value: &FFI_RangeSplitPoint) -> Result<Self, Self::Error> {
        let values = value
            .values
            .iter()
            .map(decode_scalar_value)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self::new(values))
    }
}

fn range_to_ffi(range: &RangePartitioning) -> Result<FFI_Partitioning, DataFusionError> {
    let ordering = range
        .ordering()
        .iter()
        .map(FFI_PhysicalSortExpr::from)
        .collect();
    let split_points = range
        .split_points()
        .iter()
        .map(FFI_RangeSplitPoint::try_from)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .collect();

    Ok(FFI_Partitioning::Range(ordering, split_points))
}

fn ffi_range_to_partitioning(
    ordering: &SVec<FFI_PhysicalSortExpr>,
    split_points: &SVec<FFI_RangeSplitPoint>,
) -> Result<Partitioning, DataFusionError> {
    let ordering = ordering
        .iter()
        .map(PhysicalSortExpr::from)
        .collect::<Vec<_>>();
    let ordering = LexOrdering::new(ordering).ok_or_else(|| {
        DataFusionError::Internal(
            "Range partitioning requires non-empty ordering".to_string(),
        )
    })?;
    let split_points = split_points
        .iter()
        .map(SplitPoint::try_from)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Partitioning::Range(RangePartitioning::try_new(
        ordering,
        split_points,
    )?))
}

fn encode_scalar_value(value: &ScalarValue) -> Result<SVec<u8>, DataFusionError> {
    let proto = datafusion_proto::protobuf::ScalarValue::try_from(value)
        .map_err(DataFusionError::from)?;
    Ok(proto.encode_to_vec().into_iter().collect())
}

fn decode_scalar_value(value: &SVec<u8>) -> Result<ScalarValue, DataFusionError> {
    let proto = datafusion_proto::protobuf::ScalarValue::decode(value.as_ref())
        .map_err(|err| DataFusionError::External(Box::new(err)))?;
    ScalarValue::try_from(&proto).map_err(DataFusionError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::SortOptions;
    use datafusion_common::{DataFusionError, ScalarValue};
    use datafusion_physical_expr::expressions::{Column, lit};
    use datafusion_physical_expr::{
        Partitioning, PhysicalExpr, PhysicalSortExpr, RangePartitioning, SplitPoint,
    };

    use crate::physical_expr::partitioning::FFI_Partitioning;

    #[test]
    fn round_trip_ffi_partitioning() -> Result<(), DataFusionError> {
        let col_a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let col_b = Arc::new(Column::new("b", 1)) as Arc<dyn PhysicalExpr>;
        let range_partitioning = Partitioning::Range(RangePartitioning::try_new(
            [
                PhysicalSortExpr::new(col_a, SortOptions::new(false, false)),
                PhysicalSortExpr::new(col_b, SortOptions::new(false, false)),
            ]
            .into(),
            vec![
                SplitPoint::new(vec![
                    ScalarValue::Int32(Some(10)),
                    ScalarValue::Utf8(Some("Boston".to_string())),
                ]),
                SplitPoint::new(vec![
                    ScalarValue::Int32(Some(20)),
                    ScalarValue::Utf8(None),
                ]),
            ],
        )?);

        for partitioning in [
            Partitioning::RoundRobinBatch(10),
            Partitioning::Hash(vec![lit(1)], 10),
            range_partitioning,
            Partitioning::UnknownPartitioning(10),
        ] {
            let ffi_partitioning: FFI_Partitioning = (&partitioning).into();
            let returned: Partitioning = (&ffi_partitioning).into();

            if let Partitioning::UnknownPartitioning(return_size) = returned {
                let Partitioning::UnknownPartitioning(original_size) = partitioning
                else {
                    panic!("Expected unknown partitioning")
                };
                assert_eq!(return_size, original_size);
            } else {
                assert_eq!(partitioning, returned);
            }
        }

        Ok(())
    }
}
