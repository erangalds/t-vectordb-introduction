<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$', '$'], ['\\(', '\\)']]}
  });
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Most Common Similarity Search Approaches

## Understanding the Dot Product and Similarity

The dot product of two vectors measures how much they "point" in the same direction. When two vectors are more aligned or similar in direction, their dot product is higher. Conversely, when the vectors are less aligned or orthogonal (perpendicular), their dot product is lower or zero. If they are pointing in opposite directions, the dot product can even be negative.

### Mathematical Insight

For vectors \( \mathbf{a} \) and \( \mathbf{b} \), the dot product is:



\[ \mathbf{a} \cdot \mathbf{b} = \| \mathbf{a} \| \| \mathbf{b} \| \cos(\theta) \]



Here:
- \( \| \mathbf{a} \| \) and \( \| \mathbf{b} \| \) are the magnitudes (lengths) of the vectors.
- \( \cos(\theta) \) is the cosine of the angle \( \theta \) between them.

If \( \mathbf{a} \) and \( \mathbf{b} \) point in the same direction (\( \theta = 0 \)), \( \cos(\theta) = 1 \), making the dot product \( \| \mathbf{a} \| \| \mathbf{b} \| \). If they are orthogonal (\( \theta = 90^\circ \)), \( \cos(\theta) = 0 \), and the dot product is zero.

### Example

Consider two vectors representing two documents:

\( \mathbf{d_1} = [1, 2, 3] \)

\( \mathbf{d_2} = [2, 4, 6] \)

\( \mathbf{d_3} = [1, 0, -1] \)

#### Dot Product Calculation

**Similarity Between \( \mathbf{d_1} \) and \( \mathbf{d_2} \):**



\[ \mathbf{d_1} \cdot \mathbf{d_2} = (1 \cdot 2) + (2 \cdot 4) + (3 \cdot 6) = 2 + 8 + 18 = 28 \]



**Similarity Between \( \mathbf{d_1} \) and \( \mathbf{d_3} \):**



\[ \mathbf{d_1} \cdot \mathbf{d_3} = (1 \cdot 1) + (2 \cdot 0) + (3 \cdot -1) = 1 + 0 - 3 = -2 \]



### Analysis

The dot product of \( \mathbf{d_1} \) and \( \mathbf{d_2} \) is 28, which is quite high. This high value indicates that \( \mathbf{d_1} \) and \( \mathbf{d_2} \) are very similar and point in nearly the same direction.

The dot product of \( \mathbf{d_1} \) and \( \mathbf{d_3} \) is -2, which is low and even negative. This low value indicates that \( \mathbf{d_1} \) and \( \mathbf{d_3} \) are quite dissimilar and may even point in opposite directions.

### Application in Vector Search

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

