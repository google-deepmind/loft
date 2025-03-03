# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prompt constants for SQL."""

# Fall back to another similar prompt if it is not found from the constants.
PROMPT_MAPPER = {
    "sparc": "spider",
}

############################ Corpus Instruction ################################
CORPUS_INSTRUCTION = {
    "spider": """
You will be given a list of tables. You need to memorize all of the rows of each table. Then you will be given a query, and your goal is to get the answer from the tables. Then format the answer into a list of lists. When formatting the answer into a list of lists, make sure you use the exact fields that are provided in the tables.
""".strip(),
}

CORPUS_INSTRUCTION_TEXT2SQL_BASELINE = {"spider": """
You will be given a list of SQL tables. Then you will be given a query. Your goal is to write the SQL query to get the answer from the tables. Do not include any explanation.
""".strip()}

CORPUS_FORMAT = {"spider": "Table: {title}\n{passage}"}

CORPUS_FORMAT_TEXT2SQL_BASELINE = {"spider": "{passage}"}

############################# Query Formats ####################################
QUERY_FORMAT_0 = {
    "spider": """
====== Now let's start! ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query. Then format the answer into a list of lists.
TABLES
{corpus}

Query: {query}
Answer: Here's a step-by-step approach using the provided tables:
""".strip(),
}

FOLLOW_UP_QUERY_FORMAT_0 = {
    "spider": """
====== Now let's start! ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query. Then format the answer into a list of lists.

Query: {query}

Answer:
""".strip(),
}


QUERY_NO_COT_FORMAT_0 = {
    "spider": """
====== Now let's start! ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query. Then format the answer into a list of lists.
TABLES
{corpus}

Query: {query}
Answer: Here is the final answer.
""".strip(),
}

FOLLOW_UP_QUERY_NO_COT_FORMAT_0 = {
    "spider": """
====== Now let's start! ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query. Then format the answer into a list of lists.

Query: {query}
Answer: Here is the final answer.
""".strip(),
}

QUERY_FORMAT_TEXT2SQL_BASELINE = {
    "spider": """
====== Now let's start! ======
Given the following database schema, answer the query. Do not include any explanation.

{corpus}

Query: {query}

Answer:
""".strip(),
}

FOLLOW_UP_QUERY_FORMAT_TEXT2SQL_BASELINE = {
    "spider": """
====== Now let's start! ======
Given the following database schema, answer the query. Do not include any explanation.

Query: {query}

Answer:
""".strip(),
}

################################ Few-shot Formats ##############################
FEW_SHOT_EXAMPLES_V0 = {"spider": ["""
====== Example 1 ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query. Then format the answer into a list of lists.
TABLES
Table: Concert
concert_ID,concert_Name,Theme,Stadium_ID,Year
1,Auditions,Free choice,1,2014
2,Super bootcamp,Free choice 2,2,2014
3,Home Visits,Bleeding Love,2,2015
4,Week 1,Wide Awake,10,2014
5,Week 1,Happy Tonight,9,2015
6,Week 2,Party All Night,7,2015

Table: Singer
Singer_ID,Name,Country,Song_Name,Song_release_year,Age,Is_male
1,Joe Sharp,Netherlands,You,1992,52,F
2,Timbaland,United States,Dangerous,2008,32,T
3,Justin Brown,France,Hey Oh,2013,29,T
4,Rose White,France,Sun,2003,41,F
5,John Nizinik,France,Gentleman,2014,43,T
6,Tribal King,France,Love,2016,25,T

Table: Singer_in_Concert
concert_ID,Singer_ID
1,2
1,3
1,5
2,3
2,6
3,5
4,4
5,6
5,3
6,2

Table: Stadium
Stadium_ID,Location,Name,Capacity,Highest,Lowest,Average
1,Raith Rovers,Stark's Park,10104,4812,1294,2106
2,Ayr United,Somerset Park,11998,2363,1057,1477
3,East Fife,Bayview Stadium,2000,1980,533,864
4,Queen's Park,Hampden Park,52500,1763,466,730
5,Stirling Albion,Forthbank Stadium,3808,1125,404,642
6,Arbroath,Gayfield Park,4125,921,411,638
7,Alloa Athletic,Recreation Park,3100,1057,331,637
9,Peterhead,Balmoor,4000,837,400,615
10,Brechin City,Glebe Park,3960,780,315,552

Query: Show the stadium name and the number of concerts in each stadium.

Answer: Here's a step-by-step approach using the provided tables:

**1. Access relevant data:**
We need information from two tables:
* **Concert:** This table contains details about each concert, including the stadium ID where it was held.
* **Stadium:** This table provides information about each stadium, including its name.

**2. Combine data based on stadium ID:**
We need to link the concert data with the corresponding stadium information. This can be done by joining the "Concert" and "Stadium" tables based on the common column "Stadium_ID".

**3. Get concerts per stadium:**
After joining the tables, we can group the data by stadium name and get the concerts associated with each stadium.
Here are the concert_ID associated with each stadium name:
* Stark's Park: 1
* Somerset Park: 2, 3
* Glebe Park: 4
* Recreation Park: 6
* Balmoor: 5

**Note:** Stadiums with no associated concerts are not included.

**3. Count concerts per stadium:**
After we have gotten the concerts associated with each stadium we can count how many concerts were in each stadium.
* Stark's Park: 1
* Somerset Park: 2
* Glebe Park: 1
* Recreation Park: 1
* Balmoor: 1

**4. Present the results:**
The final output will be a table showing each stadium name and the corresponding number of concerts held there.

**Based on the data provided, here's the breakdown of concerts per stadium:**

| Stadium Name | Number of Concerts |
|---|---|
| Stark's Park | 1 |
| Somerset Park | 2 |
| Glebe Park | 1 |
| Recreation Park | 1 |
| Balmoor | 1 |

Final Answer: [["Stark's Park", 1], ["Somerset Park", 2], ["Glebe Park", 1], ["Recreation Park", 1], ["Balmoor", 1]]
""".strip()]}

FEW_SHOT_EXAMPLES_V1 = {"spider": ["""====== Example 1 ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query. Then format the answer into a list of lists.
TABLES
Table: Concert
concert_ID,concert_Name,Theme,Stadium_ID,Year
1,Auditions,Free choice,1,2014
2,Super bootcamp,Free choice 2,2,2014
3,Home Visits,Bleeding Love,2,2015
4,Week 1,Wide Awake,10,2014
5,Week 1,Happy Tonight,9,2015
6,Week 2,Party All Night,7,2015

Table: Singer
Singer_ID,Name,Country,Song_Name,Song_release_year,Age,Is_male
1,Joe Sharp,Netherlands,You,1992,52,F
2,Timbaland,United States,Dangerous,2008,32,T
3,Justin Brown,France,Hey Oh,2013,29,T
4,Rose White,France,Sun,2003,41,F
5,John Nizinik,France,Gentleman,2014,43,T
6,Tribal King,France,Love,2016,25,T

Table: Singer_in_Concert
concert_ID,Singer_ID
1,2
1,3
1,5
2,3
2,6
3,5
4,4
5,6
5,3
6,2

Table: Stadium
Stadium_ID,Location,Name,Capacity,Highest,Lowest,Average
1,Raith Rovers,Stark's Park,10104,4812,1294,2106
2,Ayr United,Somerset Park,11998,2363,1057,1477
3,East Fife,Bayview Stadium,2000,1980,533,864
4,Queen's Park,Hampden Park,52500,1763,466,730
5,Stirling Albion,Forthbank Stadium,3808,1125,404,642
6,Arbroath,Gayfield Park,4125,921,411,638
7,Alloa Athletic,Recreation Park,3100,1057,331,637
9,Peterhead,Balmoor,4000,837,400,615
10,Brechin City,Glebe Park,3960,780,315,552

Query: What is the total number of singers?
Answer: Here's a step-by-step approach using the provided tables:

**1. Access relevant data:**
We need information from the "Singer" table, which stores details about each singer.

**2. Get the singers:**
We can directly count the number of rows in the "Singer" table. Each row represents a unique singer.

**Based on the data provided, the "Singer" table has the singers Joe Sharp, Timbaland, Justin Brown, Rose White, John Nizinik, and Tribal King.**

**3. Count the number of singers:**
There are 6 singers in the table.

Final Answer: [[6]]

Query: What is the name of the singer who has a song with 'Hey' in its name?
Answer: Here's a step-by-step approach using the provided tables:

**1. Identify relevant data:**
We need to look at the "Song_Name" column in the "Singer" table to find songs with "Hey" in their names.

**2. Search for matching songs:**
Scan through the "Song_Name" column and identify the song that contains "Hey."

**Based on the provided data, the song "Hey Oh" is the only one with "Hey" in its name.**

**3. Find the corresponding singer:**
Once you've identified the song, look at the "Name" column in the same row to find the singer's name.

**The singer associated with "Hey Oh" is Justin Brown.**

Final Answer: [["Justin Brown"]]

Query: Show the stadium name and the number of concerts in each stadium.
Answer: Here's a step-by-step approach using the provided tables:

**1. Access relevant data:**
We need information from two tables:
* **Concert:** This table contains details about each concert, including the stadium ID where it was held.
* **Stadium:** This table provides information about each stadium, including its name.

**2. Combine data based on stadium ID:**
We need to link the concert data with the corresponding stadium information. This can be done by joining the "Concert" and "Stadium" tables based on the common column "Stadium_ID".

**3. Get concerts per stadium:**
After joining the tables, we can group the data by stadium name and get the concerts associated with each stadium.
Here are the concert_ID associated with each stadium name:
* Stark's Park: 1
* Somerset Park: 2, 3
* Glebe Park: 4
* Recreation Park: 6
* Balmoor: 5

**Note:** Stadiums with no associated concerts are not included.

**3. Count concerts per stadium:**
After we have gotten the concerts associated with each stadium we can count how many concerts were in each stadium.
* Stark's Park: 1
* Somerset Park: 2
* Glebe Park: 1
* Recreation Park: 1
* Balmoor: 1

**4. Present the results:**
The final output will be a table showing each stadium name and the corresponding number of concerts held there.

**Based on the data provided, here's the breakdown of concerts per stadium:**

| Stadium Name | Number of Concerts |
|---|---|
| Stark's Park | 1 |
| Somerset Park | 2 |
| Glebe Park | 1 |
| Recreation Park | 1 |
| Balmoor | 1 |

Final Answer: [["Stark's Park", 1], ["Somerset Park", 2], ["Glebe Park", 1], ["Recreation Park", 1], ["Balmoor", 1]]
""".strip()]}

FEW_SHOT_EXAMPLES_TEXT2SQL_BASELINE = {"spider": ["""
====== Example 1 ======
Given the following database schema, answer the query. Do not include any explanation.

CREATE TABLE IF NOT EXISTS "stadium" (
"Stadium_ID" int,
"Location" text,
"Name" text,
"Capacity" int,
"Highest" int,
"Lowest" int,
"Average" int,
PRIMARY KEY ("Stadium_ID")
);

CREATE TABLE IF NOT EXISTS "singer" (
"Singer_ID" int,
"Name" text,
"Country" text,
"Song_Name" text,
"Song_release_year" text,
"Age" int,
"Is_male" bool,
PRIMARY KEY ("Singer_ID")
);

CREATE TABLE IF NOT EXISTS "concert" (
"concert_ID" int,
"concert_Name" text,
"Theme" text,
"Stadium_ID" text,
"Year" text,
PRIMARY KEY ("concert_ID"),
FOREIGN KEY ("Stadium_ID") REFERENCES "stadium"("Stadium_ID")
);

CREATE TABLE IF NOT EXISTS "singer_in_concert" (
"concert_ID" int,
"Singer_ID" text,
PRIMARY KEY ("concert_ID","Singer_ID"),
FOREIGN KEY ("concert_ID") REFERENCES "concert"("concert_ID"),
FOREIGN KEY ("Singer_ID") REFERENCES "singer"("Singer_ID")
);

Query: What is the total number of singers?

Answer: SELECT count(*) FROM singer

Query: What is the name of the singer who has a song with 'Hey' in its name?

Answer: SELECT Name FROM singer WHERE Song_Name LIKE '%Hey%'

Query: Show the stadium name and the number of concerts in each stadium.

Answer: SELECT stadium.name, count(*) FROM concert JOIN stadium ON concert.stadium_id = stadium.stadium_id GROUP BY concert.stadium_id
""".strip()]}

FEW_SHOT_NO_COT_EXAMPLES_V0 = {"spider": ["""====== Example 1 ======
Given a query, find from the following tables all the information relevant to the query. Then answer the query by formatting the answer into a list of lists.
TABLES
Table: Concert
concert_ID,concert_Name,Theme,Stadium_ID,Year
1,Auditions,Free choice,1,2014
2,Super bootcamp,Free choice 2,2,2014
3,Home Visits,Bleeding Love,2,2015
4,Week 1,Wide Awake,10,2014
5,Week 1,Happy Tonight,9,2015
6,Week 2,Party All Night,7,2015

Table: Singer
Singer_ID,Name,Country,Song_Name,Song_release_year,Age,Is_male
1,Joe Sharp,Netherlands,You,1992,52,F
2,Timbaland,United States,Dangerous,2008,32,T
3,Justin Brown,France,Hey Oh,2013,29,T
4,Rose White,France,Sun,2003,41,F
5,John Nizinik,France,Gentleman,2014,43,T
6,Tribal King,France,Love,2016,25,T

Table: Singer_in_Concert
concert_ID,Singer_ID
1,2
1,3
1,5
2,3
2,6
3,5
4,4
5,6
5,3
6,2

Table: Stadium
Stadium_ID,Location,Name,Capacity,Highest,Lowest,Average
1,Raith Rovers,Stark's Park,10104,4812,1294,2106
2,Ayr United,Somerset Park,11998,2363,1057,1477
3,East Fife,Bayview Stadium,2000,1980,533,864
4,Queen's Park,Hampden Park,52500,1763,466,730
5,Stirling Albion,Forthbank Stadium,3808,1125,404,642
6,Arbroath,Gayfield Park,4125,921,411,638
7,Alloa Athletic,Recreation Park,3100,1057,331,637
9,Peterhead,Balmoor,4000,837,400,615
10,Brechin City,Glebe Park,3960,780,315,552

Query: What is the total number of singers?
Answer: Here is the final answer.
Final Answer: [[6]]

Query: What is the name of the singer who has a song with 'Hey' in its name?
Answer: Here is the final answer.
Final Answer: [["Justin Brown"]]

Query: Show the stadium name and the number of concerts in each stadium.
Answer: Here is the final answer.
Final Answer: [["Stark's Park", 1], ["Somerset Park", 2], ["Glebe Park", 1], ["Recreation Park", 1], ["Balmoor", 1]]
""".strip()]}
