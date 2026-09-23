"""Glue ETL job — raw 311 NDJSON -> partitioned Parquet in the processed zone.

OPTIONAL PATH. The default Stage 1 build converts to Parquet locally
(``socrata_pull.py to-parquet``) and uses Glue only for the crawler — it is cheaper
and removes the "forgot to shut it down" risk. This job exists so the article can
show the pure-AWS path and so a reader without a local Python setup can reproduce it.

Deploy as a Glue **Spark** job (Glue 4.0, 2 DPU). Job parameters:
  --RAW_PATH        s3://<bucket>/raw/311/asof=YYYY-MM-DD/
  --PROCESSED_PATH  s3://<bucket>/processed/311/
"""

from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["JOB_NAME", "RAW_PATH", "PROCESSED_PATH"])

sc = SparkContext()
glue = GlueContext(sc)
spark = glue.spark_session
job = Job(glue)
job.init(args["JOB_NAME"], args)

TIMESTAMP_COLS = ["requested_date", "updated_date", "closed_date"]
DOUBLE_COLS = ["longitude", "latitude"]

df = spark.read.json(args["RAW_PATH"])

for col in TIMESTAMP_COLS:
    if col in df.columns:
        df = df.withColumn(col, F.to_timestamp(col))
for col in DOUBLE_COLS:
    if col in df.columns:
        df = df.withColumn(col, F.col(col).cast("double"))

df = (
    df.where(F.col("requested_date").isNotNull())
    .withColumn("year", F.year("requested_date").cast("smallint"))
    .withColumn("month", F.month("requested_date").cast("tinyint"))
    .dropDuplicates(["service_request_id"])
)

(
    df.repartition("year", "month")
    .write.mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("year", "month")
    .parquet(args["PROCESSED_PATH"])
)

print(f"wrote {df.count():,} rows to {args['PROCESSED_PATH']}")
job.commit()
