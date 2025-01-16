Euclidean Distance
Definition: Euclidean distance measures the "straight-line" distance between two points in Euclidean space. It’s the most common distance measure and is often referred to as the L2 norm.

Formula: For two vectors 
𝑎
 and 
𝑏
 in 
𝑛
-dimensional space, the Euclidean distance is:

𝑑
(
𝑎
,
𝑏
)
=
∑
𝑖
=
1
𝑛
(
𝑎
𝑖
−
𝑏
𝑖
)
2
Use Case: Often used in tasks requiring distance measurement, such as clustering and nearest neighbor search.

Intuition: It’s like the geometric distance one would measure with a ruler.

Inner Product (Dot Product)
Definition: The inner product measures the cosine of the angle between two vectors when they are normalized. It’s often used in similarity searches.

Formula: For two vectors 
𝑎
 and 
𝑏
, the inner product is:

𝑎
⋅
𝑏
=
∑
𝑖
=
1
𝑛
𝑎
𝑖
𝑏
𝑖
Use Case: Often used in machine learning for finding similarity between vectors, especially in high-dimensional spaces.

Intuition: It measures how aligned two vectors are. A higher dot product indicates higher similarity.

Key Differences
Nature of Measurement: Euclidean distance measures dissimilarity (distance), while the inner product measures similarity.

Range: Euclidean distance is always non-negative, whereas the inner product can be positive, negative, or zero.

Scale Sensitivity: Euclidean distance is sensitive to the scale of the vectors, while the inner product can be normalized to account for different vector magnitudes.

When to Use Which
Euclidean Distance: Use when you need to measure how far apart two points are in space.

Inner Product: Use when you need to measure how similar or aligned two vectors are, such as in text or document similarity.


Postgresql Function to calculate Cosine Similarity

```SQL
CREATE OR REPLACE FUNCTION cosine_similarity(vec1 vector, vec2 vector) RETURNS float8 AS $$
DECLARE
    dot_product float8 := 0;
    magnitude1 float8 := 0;
    magnitude2 float8 := 0;
BEGIN
    dot_product := vec1 <#> vec2;
    magnitude1 := sqrt(abs(vec1 <#> vec1));
    magnitude2 := sqrt(abs(vec2 <#> vec2));
    
    IF magnitude1 = 0 OR magnitude2 = 0 THEN
        RETURN 0;
    ELSE
        RETURN dot_product / (magnitude1 * magnitude2);
    END IF;
END;
$$ LANGUAGE plpgsql;

```