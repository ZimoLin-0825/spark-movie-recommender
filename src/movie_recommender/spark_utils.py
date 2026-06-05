import os
import sys

from pyspark.sql import SparkSession


def build_spark(app_name: str, master: str = "local[*]") -> SparkSession:
    """Create a local SparkSession tuned for this small project."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    spark = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
