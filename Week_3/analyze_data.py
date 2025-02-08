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
    t.title,  -- Column 'title' in title_data, not 'title_name'
    SUM(s.bandwidth) AS total_bandwidth
FROM
    streaming_data s
JOIN
    title_data t ON s.title_id = t.title_id  -- Join streaming_data with title_data
GROUP BY
    t.title  -- Group by the title name to calculate total bandwidth per title
ORDER BY
    total_bandwidth DESC  -- Sort the titles by the total bandwidth in descending order
LIMIT 10;  -- Show only the top 10 titles
"""

# Execute the query and fetch the result into a dataframe
result = conn.execute(query).fetchdf()

# Print the top 10 titles consuming the most bandwidth
print("Top 10 Titles Consuming the Most Bandwidth:")
print(result)

# Close the database connection
conn.close()

