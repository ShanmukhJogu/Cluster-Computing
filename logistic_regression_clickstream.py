"""
Logistic Regression (PySpark) - predict purchase_made from clickstream data.
Usage: python logistic_regression_clickstream.py [path/to/csv]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (StringIndexer, OneHotEncoder,
                                VectorAssembler, StandardScaler)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

PATH = (sys.argv[1] if len(sys.argv) > 1 else
        "/mnt/user-data/uploads/1793694849643_product_recommendation_clickstream_1000_rows.csv")

spark = (SparkSession.builder.appName("LogRegClickstream").master("local[*]")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.shuffle.partitions", "8").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------ 1. LOAD
df = spark.read.csv(PATH, header=True, inferSchema=True)
print("Rows:", df.count())
df.groupBy("purchase_made").count().show()

# ------------------------------------------------ 2. MISSING VALUES
cat_cols = ["product_category", "device_type", "traffic_source", "visitor_type"]
num_cols = ["session_duration_sec", "pages_viewed", "product_clicks",
            "items_viewed", "cart_additions", "search_queries",
            "avg_page_time_sec", "previous_purchases", "discount_viewed",
            "product_price", "session_hour", "day_of_week"]
missing_num = ["session_duration_sec", "pages_viewed", "avg_page_time_sec"]

# customer_id / session_id / product_id are identifiers, not features
df = df.withColumn("label", F.col("purchase_made").cast("double"))

for c in missing_num:                       # "was missing" flags
    df = df.withColumn(f"{c}_missing", F.col(c).isNull().cast("int"))
medians = {c: df.approxQuantile(c, [0.5], 0.01)[0] for c in missing_num}
df = df.fillna(medians)                     # median imputation
df = df.fillna("Unknown", subset=cat_cols)  # categorical -> "Unknown"

# product_price is right-skewed (max ~549) -> log transform
df = df.withColumn("log_price", F.log1p("product_price"))
num_cols = [c for c in num_cols if c != "product_price"] + ["log_price"]
flag_cols = [f"{c}_missing" for c in missing_num]

# ------------------------------------------------------------ 3. ENCODE
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            for c in cat_cols]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                        outputCols=[f"{c}_ohe" for c in cat_cols],
                        handleInvalid="keep")
ohe_cols = [f"{c}_ohe" for c in cat_cols]
assembler = VectorAssembler(inputCols=num_cols + flag_cols + ohe_cols,
                            outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=False, withStd=True)

# --------------------------------------------------- 4. LOGISTIC REGRESSION
lr = LogisticRegression(featuresCol="features", labelCol="label",
                        maxIter=100, regParam=0.01, elasticNetParam=0.0)
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
pred = model.transform(test).cache()

# ------------------------------------------------------------ 5. EVALUATE
tp = pred.filter("label=1 AND prediction=1").count()
tn = pred.filter("label=0 AND prediction=0").count()
fp = pred.filter("label=0 AND prediction=1").count()
fn = pred.filter("label=1 AND prediction=0").count()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
auc = BinaryClassificationEvaluator(labelCol="label").evaluate(pred)

print("\n=== Test metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")

# ------------------------------------------------ feature importance
lr_model = model.stages[-1]
names = list(num_cols) + flag_cols
for c, idx_model in zip(cat_cols, model.stages[:len(cat_cols)]):
    names += [f"{c}={l}" for l in idx_model.labels]  # OHE drops the 'invalid' slot
coefs = np.array(lr_model.coefficients)
names = names[:len(coefs)]
order = np.argsort(-np.abs(coefs))[:12]
print("\nTop coefficients (standardized features):")
for i in order:
    print(f"  {names[i]:40s} {coefs[i]:+.3f}")

# ------------------------------------------- 6. PURCHASE RATE BY CATEGORY
pred = pred.withColumn("p", vector_to_array("probability")[1])
by_cat = (pred.groupBy("product_category")
          .agg(F.avg("p").alias("predicted"), F.avg("label").alias("actual"),
               F.count("*").alias("n"))
          .orderBy(F.desc("predicted")).toPandas())
print(by_cat)

x = np.arange(len(by_cat)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, by_cat["predicted"]*100, w, label="Predicted", color="#4C78A8")
ax.bar(x + w/2, by_cat["actual"]*100, w, label="Actual", color="#F58518")
ax.set_xticks(x); ax.set_xticklabels(by_cat["product_category"], rotation=20, ha="right")
ax.set_ylabel("Purchase rate (%)")
ax.set_title("Predicted vs actual purchase rate by product category (test set)")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("purchase_rate_by_category.png", dpi=150)
print("Saved purchase_rate_by_category.png")
spark.stop()
