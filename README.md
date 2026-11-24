# Quoi de Neuf? — Khám phá & Theo dõi Chủ đề Thảo luận trên Threads

> **Mục tiêu:** Phân cụm bài viết tiếng Việt trên mạng xã hội Threads để nhận diện các chủ đề đang được quan tâm, trích xuất từ khóa đại diện cho từng cụm (`c-TF-IDF`) và đánh giá khả năng mở rộng theo dòng thời gian với Dynamic Topic Modeling (DTM).

---

## 1. Mục tiêu dự án

Trong bối cảnh nội dung trên mạng xã hội Threads ngắn, đa dạng chủ đề và chứa nhiều nhiễu ngôn ngữ, dự án tập trung xây dựng pipeline học máy cổ điển để tự động trả lời câu hỏi: ***"Trong khoảng thời gian thu thập dữ liệu, cộng đồng đang bàn tán về những chủ đề gì?"***

Trọng tâm của dự án là thiết kế một quy trình thực nghiệm chặt chẽ — từ tiền xử lý dữ liệu thô, thử nghiệm và so sánh các họ thuật toán phân cụm, trích xuất từ khóa đại diện, đến đánh giá định lượng lẫn định tính.

**Các câu hỏi nghiên cứu & kỹ thuật chính:**

1. **Biểu diễn & làm sạch văn bản:** Chiến lược tiền xử lý và vector hóa nào (ví dụ: `TF-IDF` kết hợp giảm chiều `TruncatedSVD`) xử lý hiệu quả đặc thù văn bản ngắn và nhiễu crawl trên Threads?
2. **Hiệu năng phân cụm & chất lượng chủ đề:** Giữa các họ mô hình phân cụm và không gian siêu tham số khác nhau, cấu hình nào tạo ra các cụm tách biệt nhất đồng thời cho tập từ khóa `c-TF-IDF` có tính nhất quán và độ đa dạng cao?
3. **Biến động theo thời gian *(Mở rộng)*:** Khi áp dụng góc nhìn Dynamic Topic Modeling (DTM) theo từng tuần, nội dung thảo luận bên trong cùng một chủ đề dịch chuyển như thế nào?

Dự án đánh giá kết quả trên hai nhóm bằng chứng song song:

- **Chất lượng phân cụm (Clustering Quality):** Bộ độ đo không giám sát (Silhouette Score theo khoảng cách Cosine, Davies-Bouldin Index, Calinski-Harabasz Index, Inertia) kết hợp thẩm định mẫu có giám sát trên 10% bài viết mỗi cụm.
- **Chất lượng từ khóa đại diện (Topic Quality):** Topic Coherence ($C_v$ / NPMI) và Topic Diversity (TD) trên Top-$K$ từ khóa của từng cụm.

---

## 2. Phạm vi và giao thức thực nghiệm

### 2.1. Dữ liệu đầu vào và quy tắc tiền xử lý

**Nguồn dữ liệu:** Dữ liệu dạng bảng được crawl từ Threads qua nhiều ngày, lưu tại [`data/data.csv`](data/data.csv) (hiện có **1.300 bài viết** với mốc `posted_at` từ `2026-05-25` đến `2026-11-23`).

| Cột | Kiểu / Định dạng | Mô tả |
| :--- | :--- | :--- |
| `url` | String | Đường dẫn định danh tới bài đăng Threads |
| `text` | String | Nội dung văn bản của bài viết |
| `posted_at` | ISO-8601 Timestamp | Thời điểm bài viết được đăng (dùng cho phân tích theo thời gian / DTM) |
| `collected_at` | ISO-8601 Timestamp | Thời điểm crawler thu thập dữ liệu |
| `likes` | Numeric (có thể khuyết) | Số lượt thích ghi nhận tại thời điểm crawl |
| `shares` | Numeric (có thể khuyết) | Số lượt chia sẻ ghi nhận tại thời điểm crawl |
| `comments` | Numeric (có thể khuyết) | Số lượt bình luận ghi nhận tại thời điểm crawl |

**Quy tắc tiền xử lý dữ liệu:**

- **Lọc nhiễu nội dung:** Loại bỏ các bài viết rỗng, quá ngắn không mang thông tin ngữ cảnh hoặc trùng lặp hoàn toàn.
- **Làm sạch artifact crawl & ký tự đặc biệt:** Loại bỏ các chuỗi dư thừa phát sinh từ giao diện web khi crawl (ví dụ chữ `Translate` ở cuối bài, chỉ số phân trang dạng `1 / 2`, ký tự đặc biệt/emoji không mang nghĩa chủ đề).
- **Chuẩn hóa & từ dừng (Stopwords):** Chuẩn hóa văn bản tiếng Việt, loại bỏ các từ xuất hiện phổ biến nhưng vô nghĩa về mặt phân biệt chủ đề trên Threads (ví dụ: `mọi người`, `hôm nay`, `translate`, ...).
- **Biểu diễn đặc trưng:** Sử dụng biểu diễn thưa (`TF-IDF`) kết hợp giảm chiều (như `TruncatedSVD` 50 chiều trong thử nghiệm nền) trước khi đưa vào các thuật toán phân cụm dựa trên khoảng cách.

### 2.2. Các họ mô hình và ràng buộc thuật toán

Theo yêu cầu môn học về việc sử dụng các thuật toán Machine Learning cổ điển (như K-Means, KNN, SVM, Decision Tree, ...), vai trò của từng nhóm thuật toán được tách bạch để giữ đúng bản chất bài toán không giám sát:

- **Các họ mô hình phân cụm cốt lõi (Core Clustering Models):**
  - **K-Means & K-Means++:** Mô hình baseline sử dụng tâm cụm và khoảng cách trên không gian đặc trưng đã giảm chiều / chuẩn hóa $L_2$.
  - **Các họ mô hình ứng viên cần đối chiếu:** *MiniBatchKMeans*, *Agglomerative Clustering* (phân cụm phân cấp), *DBSCAN / HDBSCAN* (phân cụm theo mật độ), *Gaussian Mixture Model (GMM)* hoặc *Spectral / k-NN Graph Clustering*. *(Danh sách model cố định cuối cùng và hyperparameter space đang được nhóm thống nhất trước khi chạy full matrix).*
- **Vai trò bổ trợ của các thuật toán có giám sát (`KNN`, `SVM`, `Decision Tree`):**
  - Không thay thế bước phân cụm khám phá chủ đề ban đầu, nhưng có thể sử dụng làm **bài kiểm tra khả năng tách biệt (Cluster Separability Check)** hoặc gán nhãn nhanh bài viết mới dựa trên tập cụm đã xác định.

### 2.3. Pipeline thiết kế thí nghiệm

Pipeline thực nghiệm được chuẩn hóa theo 6 bước tuần tự:

1. **Tiền xử lý dữ liệu:** Đọc [`data/data.csv`](data/data.csv), lọc bài thiếu ngữ cảnh, dọn artifact crawl, emoji nhiễu và stopwords.
2. **Vector hóa & Giảm chiều:** Chuyển đổi văn bản sang không gian vector (`TF-IDF` + `TruncatedSVD`) nhằm giảm độ thưa và giữ thông tin ngữ nghĩa chính.
3. **Huấn luyện & Tìm kiếm siêu tham số:** Chạy phân cụm trên các họ mô hình với từng không gian siêu tham số và dải giá trị số cụm $N$ khác nhau.
4. **Đánh giá & Chọn cấu hình tối ưu:** Chấm điểm toàn bộ cấu hình bằng bộ độ đo phân cụm không giám sát để chọn ra cấu hình tốt nhất cho mỗi họ mô hình.
5. **Trích xuất & Đánh giá từ khóa (`c-TF-IDF`):** Sinh tập Top-$K$ từ khóa đại diện cho từng cụm từ mô hình tối ưu; đánh giá bằng Topic Coherence và Topic Diversity; lặp lại bước bổ sung stopwords nếu TD thấp.
6. **Xuất kết quả & Trực quan hóa:** Lưu cấu hình/mô hình tối ưu và xuất kết quả trực quan hóa không gian cụm 2D/3D (qua giảm chiều PCA / t-SNE / UMAP) phục vụ phân tích.

### 2.4. Giao thức đánh giá (Evaluation Protocol)

#### A. Đánh giá khả năng phân cụm — Độ đo không giám sát

| Độ đo | Hướng tối ưu | Vai trò trong thực nghiệm |
| :--- | :--- | :--- |
| **Silhouette Score** *(bắt buộc `metric='cosine'`)* | Càng cao càng tốt ($\in [-1, 1]$) | Đo độ gắn kết nội cụm so với cụm lân cận gần nhất. Với văn bản ngắn và thưa trên Threads, khoảng cách Cosine phản ánh tương đồng ngữ nghĩa chính xác hơn Euclidean. *(Thử nghiệm sơ bộ trên `data.csv` với `TF-IDF` + `TruncatedSVD` 50 chiều cho thấy giá trị dao động quanh `0.02 – 0.06` do nhiễu mạng xã hội, nên chủ yếu dùng để so sánh tương đối giữa các cấu hình).* |
| **Davies-Bouldin Index (DBI)** | Càng thấp càng tốt (gần `0`) | Đo tỷ lệ trung bình giữa độ phân tán nội cụm và khoảng cách giữa tâm cụm; nhạy với các chủ đề bị chồng lấn từ vựng (ví dụ: *Sinh viên ôn thi* và *Học tiếng Anh*). |
| **Calinski-Harabasz Index (CH)** | Càng cao càng tốt | Đo tỷ lệ phương sai giữa các cụm trên phương sai nội cụm, giúp kiểm tra mức độ tách biệt về mật độ khi chia thành $N$ cụm. |
| **Inertia (WCSS)** | Giảm dần theo $N$ | Tổng bình phương khoảng cách tới tâm cụm trong **K-Means**, dùng để vẽ biểu đồ khuỷu tay (Elbow Method) xác định vùng giá trị hợp lý cho số cụm $N$. |

#### B. Đánh giá khả năng phân cụm — Kiểm chứng có giám sát *(Cần thảo luận & chốt giao thức)*

- **Ý tưởng thiết kế:** Sau khi thu được $N$ cụm từ cấu hình tối ưu, lấy ngẫu nhiên **10% số bài viết trong mỗi cụm** để nhóm thành viên gán nhãn thủ công hoặc dùng LLM đánh giá xem nội dung từng bài có khớp với chủ đề đại diện của cụm hay không.
- **Các điểm cần thống nhất trước khi thực hiện:**
  1. Chuẩn hóa thang chấm (nhị phân *khớp / không khớp* hay thang Likert 1–3) và ngưỡng tối thiểu số bài lấy mẫu đối với cụm có kích thước nhỏ.
  2. Quy định tách biệt vai trò nếu dùng LLM (tránh trường hợp cùng một LLM vừa tóm tắt tên cụm vừa tự chấm điểm độ khớp, gây thiên lệch đánh giá).
  3. Tiêu chí đánh giá độ đồng thuận (Inter-Annotator Agreement) khi có nhiều thành viên cùng kiểm tra chéo.

#### C. Đánh giá chất lượng từ khóa đại diện (`c-TF-IDF`)

- **Topic Coherence ($C_v$ hoặc NPMI — Normalized Pointwise Mutual Information):** Đo mức độ đồng xuất hiện về mặt ngữ nghĩa của Top-$K$ từ khóa (ví dụ Top-10) trong cùng một cụm trên toàn bộ tập dữ liệu. Chỉ số NPMI cao cho thấy các từ khóa thực sự mô tả một chủ đề nhất quán (ví dụ: `giáo trình`, `ôn thi`, `sinh viên`, `học kỳ`).
- **Topic Diversity (TD):** Tỷ lệ số từ khóa duy nhất trên tổng số từ khóa đại diện của tất cả $N$ cụm:

  $$\text{Topic Diversity} = \frac{\left|\bigcup_{c=1}^{N} \text{TopK}(c)\right|}{N \times K}$$

  Giá trị nằm trong $[0, 1]$ (càng gần $1.0$ càng tốt). Nếu $\text{TD} < 0.5$, nhiều cụm đang bị lặp lại cùng nhóm từ phổ biến trên Threads (như `mọi người`, `hôm nay`, `translate`) và cần quay lại bước tiền xử lý để bổ sung danh sách stopwords.

---

## 3. Hướng mở rộng: Dynamic Topic Modeling (DTM)

Ngoài bài toán phân cụm tĩnh trên toàn bộ khoảng thời gian, dự án cân nhắc mở rộng sang **Dynamic Topic Modeling (DTM)** bằng cách tận dụng trường `posted_at`. Đây là điểm khác biệt cốt lõi so với phân cụm thông thường: mô hình hóa sự dịch chuyển từ khóa và trọng tâm thảo luận trong **cùng một chủ đề** qua từng tuần.

**Ví dụ minh họa kịch bản đầu ra kỳ vọng theo tuần:**

| Chủ đề | Cuối tháng 10 (26/10 – 01/11) | Đầu tháng 11 (02/11 – 08/11) | Giữa – Cuối tháng 11 (13/11 – 23/11) |
| :--- | :--- | :--- | :--- |
| **Học tập & Sinh viên** | `giáo trình`, `ielts`, `speaking`, `giữa kỳ` | `đăng ký tín chỉ`, `thực tập`, `tiếng anh` | `20/11`, `kỷ yếu`, `studio`, `ôn thi cuối kỳ` |
| **Mua sắm & Tiêu dùng** | `pass đồ`, `thanh lý`, `máy ảnh`, `trọ` | `local brand`, `ngáo giá`, `trà sữa 711` | `black friday`, `săn sale`, `mã giảm giá` |
| **AI & Công nghệ** | `claude`, `ai agent`, `video ai`, `bản quyền` | `chatgpt pro`, `tool`, `code`, `server` | `gemini`, `tự động hóa`, `prompt`, `api` |

---

## 4. Đầu vào, đầu ra và cấu trúc repository

### Đầu vào (Inputs)

- Tập dữ liệu bài viết Threads dạng CSV tại [`data/data.csv`](data/data.csv) (hoặc JSON tại [`data/data.json`](data/data.json)).
- Danh sách từ dừng (stopwords) tùy biến cho văn bản tiếng Việt trên Threads.
- Cấu hình không gian siêu tham số và dải số cụm $N$ cho các họ mô hình phân cụm.

### Đầu ra kỳ vọng (Outputs)

- Bảng so sánh các độ đo phân cụm (Silhouette Cosine, DBI, CH, Inertia) giữa các họ mô hình và siêu tham số.
- Danh sách $N$ cụm tối ưu kèm Top-$K$ từ khóa `c-TF-IDF`, điểm Topic Coherence ($C_v$/NPMI) và Topic Diversity.
- Bảng tổng hợp kết quả thẩm định mẫu 10% (thủ công / LLM).
- File trọng số/artifact mô hình lưu tại [`saved_model/`](saved_model/README.md) và các biểu đồ trực quan hóa cụm 2D/3D.

### Cấu trúc thư mục

```text
quoi-de-neuf/
├── data/
│   ├── data.csv              # Dữ liệu bài viết Threads đã thu thập (1,300 dòng)
│   └── data.json             # Placeholder dữ liệu định dạng JSON
├── docs/
│   └── proposal.tex          # Tài liệu đề xuất / báo cáo LaTeX
├── saved_model/
│   └── README.md             # Nơi lưu trữ mô hình và artifact sau huấn luyện
├── src/
│   ├── data/                 # Module đọc và tiền xử lý dữ liệu
│   ├── pipeline/             # Pipeline vector hóa, phân cụm, trích xuất từ khóa & đánh giá
│   ├── script/
│   │   └── train.sh          # Script chạy huấn luyện thực nghiệm
│   └── notebook.ipynb        # Notebook thử nghiệm và trực quan hóa
├── .gitignore
└── README.md
```

---

## 5. Các điểm cần chốt & Tiêu chí hoàn thành

### Các quyết định cần nhóm thống nhất trước khi chạy full experiment

- [ ] Chốt danh sách chính xác các họ mô hình phân cụm sẽ đưa vào bảng so sánh (bên cạnh K-Means / K-Means++).
- [ ] Chốt giao thức đánh giá có giám sát trên 10% mẫu mỗi cụm (thang chấm, người chấm / LLM prompt, xử lý cụm nhiễu).

### Tiêu chí hoàn thành

Dự án được xem là hoàn thành khi đạt đủ các đầu mục sau:

1. **Dữ liệu sạch & tái lập được:** Quy trình tiền xử lý loại bỏ triệt để artifact crawl (`Translate`, ký tự nhiễu) và xử lý stopwords hiệu quả (TD $\ge 0.5$).
2. **So sánh mô hình đầy đủ:** Có kết quả thực nghiệm đối chiếu ít nhất 2–3 họ mô hình phân cụm với các độ đo Silhouette (`cosine`), DBI, CH và phân tích chọn số cụm $N$.
3. **Diễn giải chủ đề rõ ràng:** Trích xuất được Top-$K$ từ khóa `c-TF-IDF` kèm điểm Coherence/Diversity, kết hợp trực quan hóa 2D/3D và kiểm chứng mẫu 10%.
4. **Mở rộng *(nếu triển khai)*:** Hoàn thiện phân tích diễn biến chủ đề theo tuần (DTM) dựa trên trường `posted_at`.
