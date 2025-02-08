import duckdb as db

# Connect to the BingeBlitz.db database
conn = db.connect('BingeBlitz.db')  

# SQL query to find the titles consuming the most bandwidth


import duckdb as db

# Connect to the BingeBlitz.db database
conn = db.connect('BingeBlitz.db')  

# SQL query to find the titles consuming the most bandwidth
query = """
SELECT
    title_data.title,  -- Now using the correct 'title' column
    SUM(streaming_data.bandwidth) AS total_bandwidth
FROM
    streaming_data
JOIN
    title_data ON streaming_data.title_id = title_data.title_id
GROUP BY
    title_data.title  -- Grouping by the correct 'title' column
ORDER BY
    total_bandwidth DESC
LIMIT 10;
"""




# Execute the query and fetch the result into a dataframe
result = conn.execute(query).fetchdf()

# Print the top 10 titles consuming the most bandwidth
print("Top 10 Titles Consuming the Most Bandwidth:")
print(result)

# Close the database connection
conn.close()

