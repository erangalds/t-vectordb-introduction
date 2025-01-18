# Most Common Similarity Search Approaches
## Understanding the Dot Product and Similarity
The dot product of two vectors measures how much they "point" in the same direction. When two vectors are more aligned or similar in direction, their dot product is higher. Conversely, when the vectors are less aligned or orthogonal (perpendicular), their dot product is lower or zero. If they are pointing in opposite directions, the dot product can even be negative.

Mathematical Insight
For vectors 𝑎  and 𝑏, the dot product is:

𝑎⋅𝑏 = ∥𝑎∥∥𝑏∥cos(𝜃)

Here:
∥𝑎∥ and ∥𝑏∥ are the magnitudes (lengths) of the vectors.cos(𝜃) is the cosine of the angle 𝜃 between them.

If 𝑎 and 𝑏 point in the same direction (𝜃=0), cos(𝜃)=1, making the dot product ∥𝑎∥∥𝑏∥. If they are orthogonal (𝜃=90∘), cos(𝜃)=0, and the dot product is zero.

Example
Consider two vectors representing two documents:

𝑑1=[1,2,3]

𝑑2=[2,4,6]

𝑑3=[1,0,−1]

Dot Product Calculation
Similarity Between 
𝑑1 and 𝑑2:
𝑑1⋅𝑑2=(1⋅2)+(2⋅4)+(3⋅6)=2+8+18=28

Similarity Between 
𝑑1 and 𝑑3:
𝑑1⋅𝑑3=(1⋅1)+(2⋅0)+(3⋅−1)=1+0−3=−2

Analysis
The dot product of 𝑑1 and 𝑑2 is 28, which is quite high. This high value indicates that 𝑑1 and 𝑑2 are very similar and point in early the same direction.

The dot product of 𝑑1 and 𝑑3 is -2, which is low and even negative. This low value indicates that 𝑑1 and 𝑑3 are quite dissimilar and may even point in opposite directions.

Application in Vector Search
In vector search, we use the dot product (or cosine similarity) to determine the relevance of documents or items relative to a query vector. A higher dot product indicates that the documents or items are more similar to the query, making them more relevant.

By ranking items based on the dot product with the query vector, we can efficiently retrieve the most relevant information, enhancing search accuracy and user experience.

## Understanding the Cosine Similarity

Let's say we have two 3-dimensional vectors:
- **Vector A**: \( \mathbf{A} = [1, 2, 3] \)
- **Vector B**: \( \mathbf{B} = [4, 5, 6] \)

To calculate the cosine similarity between these vectors, we follow these steps:

1. **Calculate the dot product**:


\[ \mathbf{A} \cdot \mathbf{B} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32 \]



2. **Calculate the magnitude (norm) of each vector**:


\[ \| \mathbf{A} \| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14} \]




\[ \| \mathbf{B} \| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{16 + 25 + 36} = \sqrt{77} \]



3. **Calculate the cosine similarity**:


\[ \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\| \mathbf{A} \| \times \| \mathbf{B} \|} = \frac{32}{\sqrt{14} \times \sqrt{77}} \approx \frac{32}{\sqrt{1078}} \approx \frac{32}{32.82} \approx 0.975 \]



### Interpretation

The cosine similarity value ranges between -1 and 1:
- **1** indicates that the vectors are in the same direction.
- **0** indicates that the vectors are orthogonal (perpendicular).
- **-1** indicates that the vectors are in exactly opposite directions.

In our example, the cosine similarity of approximately 0.975 suggests that vectors A and B are quite similar in terms of direction.


## Understanding Euclidean Distance based similarity

Let's consider three 3-dimensional vectors:
- **Vector A**: \( \mathbf{A} = [1, 2, 3] \)
- **Vector B**: \( \mathbf{B} = [4, 5, 6] \)
- **Vector C**: \( \mathbf{C} = [7, 8, 9] \)

To calculate the Euclidean distance between these vectors, we follow these steps:

1. **Calculate the Euclidean distance between Vector A and Vector B**:


\[ d(\mathbf{A}, \mathbf{B}) = \sqrt{(1 - 4)^2 + (2 - 5)^2 + (3 - 6)^2} = \sqrt{(-3)^2 + (-3)^2 + (-3)^2} = \sqrt{9 + 9 + 9} = \sqrt{27} \approx 5.20 \]



2. **Calculate the Euclidean distance between Vector A and Vector C**:


\[ d(\mathbf{A}, \mathbf{C}) = \sqrt{(1 - 7)^2 + (2 - 8)^2 + (3 - 9)^2} = \sqrt{(-6)^2 + (-6)^2 + (-6)^2} = \sqrt{36 + 36 + 36} = \sqrt{108} \approx 10.39 \]



3. **Calculate the Euclidean distance between Vector B and Vector C**:


\[ d(\mathbf{B}, \mathbf{C}) = \sqrt{(4 - 7)^2 + (5 - 8)^2 + (6 - 9)^2} = \sqrt{(-3)^2 + (-3)^2 + (-3)^2} = \sqrt{9 + 9 + 9} = \sqrt{27} \approx 5.20 \]



### Interpretation

The Euclidean distance provides a measure of the straight-line distance between two vectors in the vector space. Smaller distances indicate that the vectors are closer to each other, while larger distances suggest they are farther apart.

In our example:
- The distance between Vector A and Vector B is approximately 5.20.
- The distance between Vector A and Vector C is approximately 10.39.
- The distance between Vector B and Vector C is approximately 5.20.

This suggests that Vectors A and B, and Vectors B and C, are relatively closer to each other compared to Vector A and Vector C.


