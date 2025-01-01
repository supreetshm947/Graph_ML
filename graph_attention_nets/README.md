### 1️⃣ **Node Feature Transformation**

Each node's features are linearly transformed using a weight matrix:

\[
\mathbf{h}' = \mathbf{W} \mathbf{h}
\]

Where:  
- \( \mathbf{h} \): Node feature matrix (\( N \times F_{\text{in}} \))  
- \( \mathbf{W} \): Learnable weight matrix (\( F_{\text{in}} \times F_{\text{out}} \))  
- \( \mathbf{h}' \): Transformed node feature matrix (\( N \times F_{\text{out}} \))  

In **multi-head attention**, each head learns its own weight matrix:

\[
\mathbf{h}'_i = \mathbf{W}_i \mathbf{h}
\]

Where \( i \) is the head index.

---

### 2️⃣ **Attention Mechanism**

For each edge \( (i, j) \) between nodes \( i \) (source) and \( j \) (destination), the attention score is calculated using:

\[
e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \, || \, \mathbf{W} \mathbf{h}_j \, || \, \mathbf{e}_{ij}]\right)
\]

Where:  
- \( \mathbf{a} \): Attention weight vector (\( 3 F_{\text{out}} \))  
- \( || \): Concatenation operator  
- \( \mathbf{e}_{ij} \): Edge attributes between nodes \( i \) and \( j \)  
- \( e_{ij} \): Attention score for edge \( (i, j) \)

For **multi-head attention**, each head computes:

\[
e_{ij}^{(k)} = \text{LeakyReLU}\left(\mathbf{a}_k^T [\mathbf{W}_k \mathbf{h}_i \, || \, \mathbf{W}_k \mathbf{h}_j \, || \, \mathbf{e}_{ij}]\right)
\]

---

### 3️⃣ **Attention Coefficients**

Attention scores are normalized across neighboring nodes using the **softmax** function:

\[
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(j)} \exp(e_{ik})}
\]

Where:  
- \( \mathcal{N}(j) \): Set of neighbors of node \( j \)  
- \( \alpha_{ij} \): Normalized attention coefficient  

For **multi-head attention**:

\[
\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{k \in \mathcal{N}(j)} \exp(e_{ik}^{(k)})}
\]

---

### 4️⃣ **Message Passing and Aggregation**

Node features are aggregated using the attention coefficients:

\[
\mathbf{h}_j' = \sum_{i \in \mathcal{N}(j)} \alpha_{ij} \mathbf{W} \mathbf{h}_i
\]

For **multi-head attention**, features from all heads are concatenated:

\[
\mathbf{h}_j' = ||_{k=1}^{H} \sum_{i \in \mathcal{N}(j)} \alpha_{ij}^{(k)} \mathbf{W}_k \mathbf{h}_i
\]

Where:  
- \( H \): Number of attention heads  
- \( || \): Concatenation operator  

If using **averaging** instead of concatenation (e.g., in the final layer):

\[
\mathbf{h}_j' = \frac{1}{H} \sum_{k=1}^{H} \sum_{i \in \mathcal{N}(j)} \alpha_{ij}^{(k)} \mathbf{W}_k \mathbf{h}_i
\]

---

### 5️⃣ **Final Output and Readout**

After passing through multiple GAT layers, the final node representations are aggregated across batches:

\[
\mathbf{h}_{\text{graph}} = \text{scatter}(\mathbf{h}, \mathbf{batch}, \text{reduce="mean"})
\]

Where:  
- \( \mathbf{h} \): Node embeddings after GAT layers  
- \( \mathbf{batch} \): Batch index for each node  
- \( \mathbf{h}_{\text{graph}} \): Final graph-level embedding  

---