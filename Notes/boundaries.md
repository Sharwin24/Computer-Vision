# Boundary Detection
We found components using connected component labeling. Another important feature is the boundary line or edge of the region/component. 

If segmentation is noisy, CCL won't work well because there will be several components that should make up a whole component.

## Convolution
We can use FFT on a signal to extract the frequencies from the signal. Using filters we can cutoff high/low frequencies.

3 pronged physical representation:
$$
(t-1)\,\,(t)\,\,(t+1)
$$

We can combine the function from the signal to compute the convolution:
$$
\begin{align*}
f(t-1)h(t) + f(t)h(0) + f(t+1)h(t) \\
\frac{f(t-1) + f(t) + f(t+1)}{3}
\end{align*}
$$

Expressed in integral form across the signal $h(t)$.
$$
\int_n x(t-n)h(n)\,dn = y(t) \to \int x(n)h(n-t)\,dn
$$

## Preliminaries (Set Theory)

$$
A \cap B \to \text{intersection} \\
A \cup B \to \text{union} \\
A^c \to \text{complement} \\
A - B = \{w \mid w \in A, w \notin B \} = A \cap B^c \to \text{difference} \\
\hat{B} = \{w \mid w = -b, \forall b \in B\} \to \text{reflection} \\
?'A \to \text{translation}
$$

Dilation can bridge gaps using a kernel across a segmented image that has gaps in components where we don't expect them.

$$
A \oplus B = \left\{z \mid \left(\hat{B}\right)_z \cap A \neq \phi \right\} = \cup_{a_i \in A} B_{a_i}
$$

Erosion can eliminate extra components to reduce the size of a boundary.

$$
A \Theta B = \left\{z \mid B_z \subseteq A \right\}
$$

Question: $\left(A \Theta B \right) \subseteq A ???$

Opening and Closing can only be applied once. Hit-or-Miss transformation is a basic tool for shape detection that finds the location of a particular shape.

Optical Character Recognition (OCR).
- Assume characters are of the same font and size
Model Construction:
- Extract character to be recognized
- Use opening and closing to fill holes and cavities
- Shrink the character image to remove unwanted region to reduce the size, s.t. it fits inside an instance of the character

Boundary Extraction: Find the line/edge that wraps around a component. In Binary image, this can be defined as the component pixels next to background pixels so the boundary is 1 pixel wide.

$$
A - (A \ominus B) = \underbrace{\Beta(A)}_{\text{Boundary of A}}
$$

**Computational issue:**

if $B = B_1 \oplus B_2 \oplus \cdots \oplus B_k$

then $A \oplus B = \left\{\left(A \oplus B_1\right) \oplus B_2 \oplus \cdots \oplus B_k \right\}$

If we decompose the dilations in different directions we can save computation since we are commuting all dilations together.
