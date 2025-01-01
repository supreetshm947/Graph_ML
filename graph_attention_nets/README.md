### 1️⃣ **Node Feature Transformation**

Each node's features are linearly transformed using a weight matrix:

**h' = W * h**

Where:  
- **h**: Node feature matrix (N × F_in)  
- **W**: Learnable weight matrix (F_in × F_out)  
- **h'**: Transformed node feature matrix (N × F_out)  

For **multi-head attention**, each head learns its own weight matrix:

**h'_i = W_i * h**

Where **i** is the head index.

---

### 2️⃣ **Attention Mechanism**

For each edge `(i, j)` between nodes `i` (source) and `j` (destination), the attention score is calculated using:

**e_ij = LeakyReLU(a^T [W * h_i || W * h_j || e_ij])**

Where:  
- **a**: Attention weight vector (3 * F_out)  
- `||`: Concatenation operator  
- **e_ij**: Edge attributes between nodes `i` and `j`  
- **e_ij**: Attention score for edge `(i, j)`

For **multi-head attention**, each head computes:

**e_ij^(k) = LeakyReLU(a_k^T [W_k * h_i || W_k * h_j || e_ij])**

---

### 3️⃣ **Attention Coefficients**

Attention scores are normalized across neighboring nodes using the **softmax** function:

**α_ij = exp(e_ij) / Σ_{k ∈ N(j)} exp(e_ik)**

Where:  
- **N(j)**: Set of neighbors of node `j`  
- **α_ij**: Normalized attention coefficient  

For **multi-head attention**:

**α_ij^(k) = exp(e_ij^(k)) / Σ_{k ∈ N(j)} exp(e_ik^(k))**

---

### 4️⃣ **Message Passing and Aggregation**

Node features are aggregated using the attention coefficients:

**h_j' = Σ_{i ∈ N(j)} α_ij * W * h_i**

For **multi-head attention**, features from all heads are concatenated:

**h_j' = ||_{k=1}^H Σ_{i ∈ N(j)} α_ij^(k) * W_k * h_i**

Where:  
- **H**: Number of attention heads  
- `||`: Concatenation operator  

If using **averaging** instead of concatenation (e.g., in the final layer):

**h_j' = (1 / H) * Σ_{k=1}^H Σ_{i ∈ N(j)} α_ij^(k) * W_k * h_i**

---

### 5️⃣ **Final Output and Readout**

After passing through multiple GAT layers, the final node representations are aggregated across batches:

**h_graph = scatter(h, batch, reduce="mean")**

Where:  
- **h**: Node embeddings after GAT layers  
- **batch**: Batch index for each node  
- **h_graph**: Final graph-level embedding  

---
