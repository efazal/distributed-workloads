from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Array, String

# Define your entity (primary key for feature lookup)
wiki_passage = Entity(
    name="wiki_passage",
    join_keys=["passage_id"],
    value_type=ValueType.INT64,
    description="Unique ID of a Wikipedia passage",
)
# Define offline source
wiki_dpr_source = FileSource(
    name="wiki_dpr_source",
    path="data/wiki_dpr.parquet",
    timestamp_field="event_timestamp",
)


# Define the feature view for the Wikipedia passage content
wiki_passage_content_fv = FeatureView(
    name="wiki_passage_content",
    entities=[wiki_passage],
    ttl=timedelta(days=1),
    schema=[
        Field(
            name="text",
            dtype=String,
            description="Content of the Wikipedia passage"
        ),
        Field(
            name="embedding", 
            dtype=Array(Float32),
            description="vectors"
        )
    ],
    online=True,
    source=wiki_dpr_source,
    description="Content features of Wikipedia passages",
)

# Define the feature view for the pre-computed embeddings
wiki_passage_embeddings_fv = FeatureView(
    name="wiki_passage_embeddings",
    entities=[wiki_passage],
    ttl=timedelta(days=1),
    schema=[
        Field(
            name="vector",
            dtype=Array(Float32),
            description="Pre-computed embedding vector of the passage"
        ),
    ],
    online=True,
    source=wiki_dpr_source,
    description="Pre-computed embedding features of Wikipedia passages",
)