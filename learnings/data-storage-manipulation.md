# Data Storage & Manipulation

## AWS Resources

### S3

- Typically used as first landing spot for raw data
  - Decouples data producers from main read store.
  - Cheap, durable, scalable storage. Cold storage can be used here (though most AWS storage resources have some form of this)
  - Enables reprocessing of raw data. I've definitely learned this lesson the hard way a few times
- I saw AWS offers some robotics-centric features like [this](https://aws.amazon.com/blogs/robotics/record-store-robot-data-rosbag/), but don't see myself getting into that anytime soon
- For this implementation I decided on the following:
  - Use just a single bucket to store all the data for a type of sensor.
  - Use **Hive-style Partitioning**:
    - Organize data using prefixes formated as `key=value/` (e.g. `year=YYYY/month=MM/day=DD/hour=HH/sensor_id=UUID`
  - Keep the writer simple, let the writing application determine this partition key. TODO
    - Same with batch size and all that probably

### Timestream

- AWS' wrapper on InfluxDB
- Time-series database
- Best for large-scale ingestion and querying
- Also offers cold storage (called magnetic)

## Data Storage Formats

### Apache Parquet

- columnar-store file format
- best for large datasets (like time-series!)
- Great [tl;dr](https://www.youtube.com/watch?v=PaDUxrI6ThA)
- Can get varied with how to use Parquet, but with this project we're going with the method of just writing the entire batch in context to a single file
  - Maybe when the transformers library is more built out we can get into bucketizing the data based on some window size (or even memory size?)
- Files are IMMUTABLE
- Got curious about how folks address the "challenges" of Parquet and found out about [Apache Iceberg](https://www.youtube.com/watch?v=TsmhRZElPvM) which I might explore later

## Data Movement & Processing

### Apache Arrow

- "The universal columnar format and multi-language toolbox for fast data interchange and in-memory analytics"
- The use cases I've seen mostly involve improving interoperability between different data processing tools (Python, Spark, Pandas, Polars, DBs, etc.)
  - Does this by making in-memory management of (columnar store) large datasets incredibly efficient
- One [talk](https://www.youtube.com/watch?v=Hqi_Bw_0y8Q) I enjoyed
