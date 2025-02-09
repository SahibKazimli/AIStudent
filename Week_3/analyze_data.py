
# SQL query to find the titles consuming the most bandwidth
import duckdb as db

# Connect to the BingeBlitz.db database
conn = db.connect('BingeBlitz.db')  

# Connect to the BingeBlitz.db database
conn = db.connect('BingeBlitz.db')  

# SQL query to find the titles consuming the most bandwidth
join_query = """
SELECT
    title_data.title,  
    SUM(streaming_data.bandwidth) AS total_bandwidth
FROM
    streaming_data
JOIN
    title_data ON streaming_data.title_id = title_data.title_id
GROUP BY
    title_data.title  
ORDER BY
    total_bandwidth DESC
LIMIT 10;
"""

usage_query = """
SELECT 
    streaming_data.region,
    EXTRACT(HOUR FROM time_measured) AS hour_of_day,
    COUNT(*) AS total_streams
FROM
    streaming_data
GROUP BY 
    region, hour_of_day
ORDER BY
    region, total_streams DESC
LIMIT 10;
"""

resolution_query = """
WITH total_streams AS (
    -- Get total amount of streams
    SELECT COUNT(*) as total_count
    FROM streaming_data
)
SELECT 
    resolution,
    COUNT(*) AS resolution_count,
    COUNT(*) * 100/total_count AS resolution_percentage  -- Calculate percentage
FROM 
    streaming_data,
    total_streams -- Join with the total count
    
GROUP BY
    resolution, total_count -- 
ORDER BY
    resolution_count DESC;
"""

device_query = """

WITH total_devices AS(
    -- Get total amount of devices
    SELECT COUNT(*) AS total_device_count
    FROM streaming_data
)

SELECT 
    device,
    COUNT(*) AS device_count,
    COUNT(*) * 100 / total_device_count AS device_percentage -- Calculate percentage
FROM 
    streaming_data, 
    total_devices -- Join with total devices amount
GROUP BY
    device, total_device_count
ORDER BY 
    device_count DESC;
"""


# Execute the query and fetch the result into a dataframe
result = conn.execute(join_query).fetchdf()
peak_usage_times = conn.execute(usage_query).fetchdf()
common_resolution = conn.execute(resolution_query).fetchdf()
common_devices = conn.execute(device_query).fetchdf()



# Print the top 10 titles consuming the most bandwidth
print("Top 10 Titles Consuming the Most Bandwidth:")
print(f"{result}\n")
print(f"{peak_usage_times}\n")
print(f"{common_resolution}\n")
print(f"{common_devices}\n")

# Close the database connection
conn.close()

