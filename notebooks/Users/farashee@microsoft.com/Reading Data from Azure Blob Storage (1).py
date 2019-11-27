# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This notebook shows you how to create and query a table or DataFrame loaded from data stored in Azure Blob storage.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1: Set the data location and type
# MAGIC 
# MAGIC There are two ways to access Azure Blob storage: account keys and shared access signatures (SAS).
# MAGIC 
# MAGIC To get started, we need to set the location and type of the file.

# COMMAND ----------

storage_account_name = "fs.azure.account.key.<YOUR-BLOB-ACCOUNT-NAME>.blob.core.windows.net"
storage_account_access_key = "<YOUR-ACCESS-KEY>"

# COMMAND ----------

file_location = "wasbs://<YOUR-BLOB-ACCOUNT-NAME>.blob.core.windows.net/<YOUR-FILE-NAME>"

# COMMAND ----------

spark.conf.set(
  storage_account_name,
  storage_account_access_key)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Read the data
# MAGIC 
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

df = spark.read.format('csv').options(header='true', inferSchema='true').load(file_location).withColumnRenamed("home.dest", "home_dest")

# COMMAND ----------

df = df.na.fill({'age': 0, 'fare': 0})
df = df.withColumn("age_num", df.age.cast('float')).drop('age')
df = df.withColumn("fare_num", df.fare.cast('float')).drop('fare')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Query the data
# MAGIC 
# MAGIC Now that we have created our DataFrame, we can query it. For instance, you can identify particular columns to select and display.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT name, sex, age_num, survived 
# MAGIC FROM titanic

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4: Save as Databricks Table
# MAGIC 
# MAGIC Saving as a databricks table

# COMMAND ----------

df.write.format("csv").mode("overwrite").saveAsTable("titanic")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This table will persist across cluster restarts and allow various users across different notebooks to query this data.

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(*) from titanic

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from titanic

# COMMAND ----------

