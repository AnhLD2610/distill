import pyarrow.parquet as pq

path = "data.parquet"
pf = pq.ParquetFile(path)

num_rows = pf.metadata.num_rows
print("num_rows =", num_rows)