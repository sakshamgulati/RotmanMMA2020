// Databricks notebook source
val x = spark.sql("select * from game_CSV")
x.filter(x("venue_time_zone_id")==="America/New_York").show()

// COMMAND ----------

sql("show functions").show()

// COMMAND ----------

sql("show tables").show()

// COMMAND ----------

sql("select * from game_csv").show(5)

// COMMAND ----------

// MAGIC %sql
// MAGIC show tables

// COMMAND ----------

// MAGIC %sql
// MAGIC select distinct a.shortName from team_info_csv as a inner join game_csv as b on (a.team_id=b.away_team_id)

// COMMAND ----------

sql("").show()
