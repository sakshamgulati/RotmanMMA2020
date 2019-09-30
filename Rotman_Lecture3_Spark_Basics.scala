// Databricks notebook source
//Lecture 3 notebook
//Covering basics of spark 


// COMMAND ----------

// lets create a data frame by manually entering all values 
val training = spark.createDataFrame(Seq(
  (0L, "a b c d e spark", 1.0),
  (1L, "b d", 0.0),
  (2L, "spark f g h", 1.0),
  (3L, "hadoop mapreduce", 0.0)
)).toDF("id", "text", "label")

// COMMAND ----------

//writing down files in HDFS 
training.write.format("csv").option("path", "/FileStore/tables/")

// COMMAND ----------

//lets list all tables 
sql("show tables").show()          

// COMMAND ----------

//Lets use excel files we have uploaded into our database 

//lets create a dataframe by uploading game table 
val game = spark.table("game_csv")

// new data frame has to always be declared with val 
val game2 = game

// COMMAND ----------

//lest try most basic operations on dataframes 

//you can apply actions on decrated dataframes 
//look into each of the functions to undrstand what they do

//game.show(10)
//game.schema
game.printSchema()
//game.dtypes(0)


// COMMAND ----------

//SELECT 

//Select is used to select specific columns from a dataframe
val test  = game.select("game_id", "season")


// COMMAND ----------

//Order and Group by 

//Order records by 'season'
//game.orderBy($"season").show()

// Group by
//find max away score for each season
import org.apache.spark.sql.functions._
game.groupBy("season").agg(max("away_goals")).show

// COMMAND ----------

//FILTER statement 

//list all games in season 20122013
game.filter ($"season" === 20122013).show()

//find games where away scores>2
game.filter($"away_goals">2).show()

// COMMAND ----------

//FILTER statement 

// example, apply two conditions 
//count all games where away_goals >5 and home_goals>5

val df1 = sql("select * from game_csv")
val df2 = df1.filter($"away_goals" >5)
val df3 = df2.filter($"home_goals" >5)

df3.count()


// COMMAND ----------

//FILTER statement 

//apply two conditions (AND) 

//find games where away_goals>2 and home_goals>5
val game3 = game.filter($"away_goals">2 && $"home_goals">2).show

//Apply two conditions (OR)

//find all games that where away_goals=6 or home_goals=6
val game4 = game.filter($"away_goals"===6 || $"home_goals"===6).show

// COMMAND ----------

//distinct function 
//Count number of distinct venues in game_CSV 
val test =game.select("venue").distinct().count()

// COMMAND ----------

// add a new column with withColumn function 
//import necessary/additional libraries 
import org.apache.spark.sql.functions._

//add a column that copies another column 
//game.select("venue").withColumn("Test", expr("venue")).show()

//add total score column Total score that sums values from two other columns
val game3 = game.select("home_goals", "away_goals").
withColumn("Total_score", expr("home_goals+away_goals"))

//lets add a new column with all values equal 10 
game.withColumn("Test", lit(10)).show(2)

// COMMAND ----------

//Lazy transformations vs actions

// first 3 steps are lazy, count is an action 

val df_test = spark.table("game_csv")

val df_test1 = df_test.select("game_id","away_goals")

val df_test2 = df_test1.filter($"away_goals">2)
df_test2.count()

// COMMAND ----------

//Chaining of actions/transformation in Spark 
val df_test = spark.table("game_csv")
val df_test1 = df_test.select("game_id","away_goals")
val df_test2 = df_test1.filter($"away_goals">2)
df_test2.count


// It can be rewritten as: 
val df_test = spark.table("game_csv").select("game_id","away_goals").filter($"away_goals">2).count()

// COMMAND ----------

//Describe and explain 

val df_test = spark.table("game_csv")
val df_test1 = df_test.select("game_id","away_goals").filter($"away_goals">2)
df_test2.explain()
df_test2.describe().show()


// COMMAND ----------

//working with nulls with na.drop()

//The default is to drop any row in which any value is null:
val test  = game.na.drop()
test.count()

// COMMAND ----------

//filter records using WHERE statement 
val game_4 = game.where(($"venue"==="TD Garden")||($"venue"==="Madison Square"))

// COMMAND ----------

//To be completed by students  - practice groupBy

//Task
//For every season count number of games
//With gome goals > 2 and
//With away goals > 2
//Apply groupBy().agg(count) syntaxis

import org.apache.spark.sql.functions.count

//Finish this on your own 
val df = spark.table(

val df1 = df.select

val df2  = df1.filter

val df3 = df2.groupBy

game.count()

// COMMAND ----------

//Lets use WHEN -OTHERWISE to populate a new column 

val game4 = game.withColumn("VenueUpdated", when($"venue".isin("TD Garden","Boston"), "Boston_Venue").otherwise("Other"))

game4.select("venue", "VenueUpdated").show()


// COMMAND ----------

//more complicated operations

// COMMAND ----------

//Append two tables to each other 
import org.apache.spark.sql.functions._

val x = game.union(game).union(game)
x.count()

// COMMAND ----------

//Simple Join 

val game = spark.table("game_csv")
val team_info = spark.table("team_info_csv")


val df = team_info.join(game, team_info.col("team_id") === game.col("away_team_id"))

//df_game.count()
//df_team_info.count()
df.count()

// COMMAND ----------

//JOIN 
// find all games where away_goal>6 and list names of teams that scored as away_team 

val df_game = spark.table("game_csv").filter($"away_goals">6)

val df_team_info = spark.table("team_info_csv")

val df = df_team_info.join(df_game, df_team_info.col("team_id") === df_game.col("away_team_id")).
select(df_team_info.col("team_id"),df_team_info.col("teamName"),df_game.col("away_goals")).
show()


// COMMAND ----------

//to be completed by students 

// COMMAND ----------

//Step 1: 
//count number of records in all 3 tables 


// COMMAND ----------

//Step 2: 
//select season, game_id and venue from game_CSV, order by venue and print 



// COMMAND ----------

//Step 3: 
//Count names of all head_coaches and count how many of them are on the list



// COMMAND ----------

//Step 4
//Count all games where that happened at TD Garden and away_goals >5




// COMMAND ----------

//Step 6: to be completed by students 
//find all games that happened during the seasons 20122013 and 20132014

//count number of games that happened during those two seasons


// COMMAND ----------

//Step 7:  

//find max away, min away, average away, max home, min home, average home scores for each venue


// COMMAND ----------

//Step: 8 
//how many records have been created when you append game to itself? (use count function)



// COMMAND ----------

//Step 9:

//List names of teams that played as away team at TD garden center
//How many teams are on the list? 



// COMMAND ----------

val team_info_CSV = spark.table("team_info_csv")
val game_CSV = spark.table("game_csv")

val df1 = team_info_CSV.join(game_CSV, team_info_CSV.col("team_id") === game_CSV.col("away_team_id")).filter($"venue" === "TD Garden")
df1.printSchema()
df1.select("shortName").distinct().count()
