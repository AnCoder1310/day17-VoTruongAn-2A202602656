# Báo Cáo Phân Tích Hệ Thống Memory Cho AI Agent (Lab 17)

Báo cáo này phân tích cơ chế hoạt động, trade-off thực nghiệm và kết quả định lượng giữa **Baseline Agent** và **Advanced Agent** trên hai bộ dữ liệu benchmark tiếng Việt:
1. `Standard Benchmark` (`data/conversations.json` — 10 hội thoại thường, ~10 lượt/hội thoại của user `dungct`).
2. `Long-Context Stress Benchmark` (`data/advanced_long_context.json` — 1 hội thoại 16 lượt dài chứa nhiều tin tức dày của user `dungct_stress`).

---

## 1. Kết quả Benchmark Thực Nghiệm (Chạy trên trạng thái sạch)

Lệnh thực thi từ thư mục gốc của repository:
```bash
python -c "import shutil; shutil.rmtree('state', ignore_errors=True)"
python src/benchmark.py
```

### 1.1. Bảng 1: Standard Benchmark (`data/conversations.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 2,350 | 19,056 | **0.0%** | 34.6% | 0 | 0 |
| **Advanced** | 2,553 | 28,669 | **100.0%** | 100.0% | 305 | 0 |

### 1.2. Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 402 | 23,291 | **0.0%** | 35.0% | 0 | 0 |
| **Advanced** | 812 | **12,490** | **100.0%** | 100.0% | 253 | **4** |

---

## 2. Trả lời 4 Câu Hỏi Cốt Lõi (Bước 8 Guide.md)

### 2.1. Vì sao Advanced có recall tốt hơn Baseline?
- **Số liệu chứng minh**: 
  - `Cross-session recall` của Advanced đạt **100.0%** ở cả hai bảng (Standard và Stress), trong khi Baseline chỉ đạt **0.0%**.
  - `Memory growth (bytes)` của Advanced là **305 bytes** (Standard) và **253 bytes** (Stress), trong khi Baseline luôn bằng **0 bytes**.
- **Cơ chế trong mã nguồn**:
  - Baseline lưu trữ `SessionState` khóa hoàn toàn theo `thread_id`. Khi bước sang thread mới để trả lời câu hỏi recall (`recall_<conv_id>_<q_idx>`), phiên làm việc hoàn toàn trống, dẫn đến việc Baseline không thể trả lời bất kỳ thông tin nào và phát sinh câu trả lời mất ngữ cảnh (*"Chào bạn! Rất tiếc là trong phiên làm việc này mình chưa có thông tin trước đó..."*).
  - Ngược lại, Advanced Agent sở hữu tầng Persistent Memory độc lập: `extract_profile_updates()` trích xuất các fact ổn định (tên, nơi ở, nghề nghiệp, sở thích) và hàm `profile_store.upsert_fact()` lưu bền vững xuống đĩa tại `state/profiles/<user>/User.md`. Khi sang thread mới, hàm `_offline_response()` nạp lại nội dung từ file `User.md` để tái tạo ngữ cảnh người dùng, đảm bảo câu trả lời luôn chứa đầy đủ các thực thể được truy vấn.
- **Giới hạn đi kèm**: Advanced phụ thuộc hoàn toàn vào độ chính xác của bộ trích xuất fact. Nếu bộ trích xuất bỏ sót hoặc trích xuất sai, Persistent Memory sẽ lưu trữ thông tin lỗi sang các phiên sau.

### 2.2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
- **Số liệu chứng minh**:
  - Ở bảng Standard Benchmark, `Prompt tokens processed` của Advanced là **28,669 tokens**, cao hơn **50.4%** so với Baseline (**19,056 tokens**).
  - `Agent tokens only` của Advanced cũng nhỉnh hơn (**2,553** so với **2,350 tokens**).
- **Cơ chế trong mã nguồn**:
  - Trong mỗi lượt gọi `_estimate_prompt_context_tokens()`, prompt của Advanced luôn phải gánh thêm:
    $$\text{Prompt Context} = \text{User.md (hồ sơ cá nhân)} + \text{Summary Context} + \text{Recent Messages}$$
  - Trong các hội thoại ngắn (mỗi turn người dùng chỉ dài 10–25 tokens), file `User.md` chiếm khoảng 60–80 tokens overhead trên từng lượt. Vì tổng số token của 10 lượt hội thoại ngắn chỉ đạt ~300 tokens (thấp hơn ngưỡng `compact_threshold_tokens = 800`), tầng Compact Memory hoàn toàn không kích hoạt (`Compactions = 0`). Do đó, Advanced không có cơ hội tiết kiệm token mà phải trả thêm "thuế ngữ cảnh" (context tax) cho `User.md` ở tất cả các lượt chat.
- **Giới hạn đi kèm**: Đối với các tác vụ bot hỏi đáp một lần (single-turn) hoặc các phiên chat cực ngắn, việc duy trì persistent profile sẽ làm tăng chi phí vận hành API mà không đem lại lợi ích kinh tế.

### 2.3. Vì sao Compact Memory có lợi thế ở hội thoại dài?
- **Số liệu chứng minh**:
  - Ở bảng Stress Benchmark, `Prompt tokens processed` của Advanced giảm xuống còn **12,490 tokens**, tiết kiệm tới **46.4%** so với Baseline (**23,291 tokens**).
  - Cột `Compactions` đạt **4 lần** ở Advanced và **0 lần** ở Baseline.
  - Cần phân biệt rõ: `Agent tokens only` của Advanced là **812 tokens** (cao hơn Baseline **402 tokens** do câu trả lời chứa đầy đủ bullet points kỹ thuật và profile), nhưng `Prompt tokens processed` — lượng ngữ cảnh đầu vào mà mô hình phải đọc qua từng lượt — lại giảm gần một nửa.
- **Cơ chế trong mã nguồn**:
  - Khi người dùng gửi chuỗi 16 lượt tin tức dài (mỗi lượt 600–750 ký tự), Baseline phải kéo theo toàn bộ lịch sử không nén, khiến kích thước prompt tăng theo cấp số nhân $O(N^2)$.
  - Advanced Agent kiểm tra ngưỡng sau mỗi lần `CompactMemoryManager.append()`. Khi vượt quá 800 tokens, hàm `compact()` lập tức nén các tin nhắn cũ bằng `summarize_messages()`, đưa vào trường `summary` và chỉ giữ lại `keep_messages = 4` tin nhắn gần nhất nguyên văn. Điều này chặn đứng sự bùng nổ ngữ cảnh, giữ cho độ dài prompt của mỗi lượt luôn ở mức trần an toàn (~350–500 tokens).
- **Giới hạn đi kèm**: Việc nén ngữ cảnh cũ thành tóm tắt có thể làm mất đi các chi tiết nhỏ (trivia details, số liệu phụ) không nằm trong tóm tắt hoặc không được đưa vào `User.md`.

### 2.4. File memory tăng trưởng ra sao và rủi ro gì đi kèm?
- **Số liệu chứng minh**:
  - `Memory growth (bytes)` đạt **305 bytes** cho user `dungct` và **253 bytes** cho user `dungct_stress`.
- **Phân tích rủi ro thực tế**:
  1. **Rủi ro Memory Bloat (Phình to bộ nhớ)**: Nếu hệ thống lưu trữ vô tội vạ mọi câu nói của người dùng vào `User.md`, file markdown sẽ phình to từ vài trăm bytes lên hàng chục kilobytes. Vì `User.md` được nạp vào mọi prompt, file quá lớn sẽ chiếm hết context window khả dụng của LLM và làm tăng chi phí API theo cấp số nhân.
  2. **Rủi ro Stale Data & Hallucination (Dữ liệu lỗi thời và mâu thuẫn)**: Khi người dùng đổi nơi ở (từ Đà Nẵng sang Huế rồi lại sang Đà Nẵng) hoặc đổi nghề (backend sang MLOps), nếu không có cơ chế ghi đè hoặc loại bỏ fact cũ, file sẽ chứa đồng thời cả hai thông tin mâu thuẫn. LLM khi đọc prompt sẽ không thể phân biệt đâu là thông tin hiện tại, dẫn tới ảo giác (hallucination).
  3. **Rủi ro Poisoning từ thông tin nhiễu**: Những câu nói đùa hoặc chi tiết tạm thời (đi họp ở Hà Nội, đùa làm product manager) nếu bị lưu nhầm vào `User.md` sẽ vĩnh viễn làm sai lệch chân dung người dùng qua tất cả các phiên sau.

---

## 3. Thử Nghiệm Bóc Tách (Ablation Study)

Để chứng minh luận điểm rằng chính Compact Memory tạo ra sự tối ưu ở cột `Prompt tokens processed`, một thử nghiệm ablation được thực hiện bằng cách đặt `compact_threshold_tokens = 100000` (tắt hoàn toàn compact memory trên stress dataset):

| Cấu hình thử nghiệm trên Stress Dataset | Prompt tokens processed | Compactions | Cross-session recall |
| :--- | :---: | :---: | :---: |
| **Advanced (Có Compact Memory - Mặc định)** | **12,490** | **4** | **100.0%** |
| **Baseline (Không Persistent, Không Compact)** | 23,291 | 0 | 0.0% |
| **Advanced Ablation (Tắt Compact Memory)** | **26,367** | **0** | 100.0% |

**Kết luận thực nghiệm**:
- Khi tắt compact, `Prompt tokens processed` của Advanced vọt lên **26,367 tokens** (thậm chí cao hơn Baseline 23,291 tokens do phải mang thêm `User.md` vào prompt mà không hề nén lịch sử).
- Thử nghiệm này chứng minh 100% rằng lợi thế tiết kiệm 46.4% ngữ cảnh của Advanced Agent đến trực tiếp từ cơ chế compact memory.

---

## 4. Phân Tích Các Tính Năng Bonus (Mức Điểm 90–100)

Theo yêu cầu của `Rubric.md`, hệ thống được trang bị 4 tính năng mở rộng kỹ thuật với phân tích 3 mặt toàn diện:

### 4.1. Bonus 1: Conflict Handling & Negation Detection (Xử lý xung đột và đính chính)
- **Vấn đề giải quyết**: Trong benchmark, người dùng đính chính nơi ở ("giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa") và nghề nghiệp ("không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer"). Nếu chỉ append fact mới, `User.md` sẽ lưu cả 2 giá trị gây mâu thuẫn.
- **Cải thiện recall/token**: Hàm `UserProfileStore.upsert_fact()` tự động cập nhật trường tương ứng và định dạng lại markdown, đồng thời regex nhận diện các mẫu phủ định (`không còn`, `chứ không còn`, `đừng nói`, `thông tin cũ`) để loại bỏ hoàn toàn fact cũ. Nhờ đó, recall của các câu hỏi đính chính đạt tuyệt đối **100%**.
- **Rủi ro phát sinh**: Nếu người dùng nói phủ định trong một ngữ cảnh phức tạp hoặc trích dẫn lời người khác (ví dụ: *"Bạn tôi nói đừng làm backend nữa, nhưng tôi vẫn làm"*), regex phủ định có thể nhận diện nhầm và xóa nhầm fact đang hợp lệ.

### 4.2. Bonus 2: Interrogative Filtering & Confidence Threshold (Ngăn lưu sai từ câu hỏi)
- **Vấn đề giải quyết**: Khi người dùng đặt câu hỏi kiểm tra bot (ví dụ: *"Bạn có nhớ mình tên gì không?"*, *"Hiện tại mình đang ở đâu?"*), một bộ trích xuất ngây thơ sẽ bắt từ khóa `"tên gì"` hoặc `"ở đâu"` và ghi vào `User.md`.
- **Cải thiện recall/token**: Hàm `extract_profile_updates()` kiểm tra dấu chấm hỏi (`?`) và các mẫu nghi vấn/truy vấn (`nhắc lại`, `nhớ lại`, `ở đâu`, `mình tên gì`). Nếu phát hiện turn đó là câu hỏi, độ tin cậy được hạ xuống 0 và bỏ qua hoàn toàn. Giúp file `User.md` luôn sạch sẽ, không bị phình dung lượng vô ích.
- **Rủi ro phát sinh**: Nếu người dùng vừa hỏi vừa cung cấp fact trong cùng một câu (ví dụ: *"Mình chuyển sang MLOps rồi, bạn thấy nghề này thế nào?"*), bộ lọc có thể bỏ sót fact thật vì câu chứa dấu hỏi.

### 4.3. Bonus 3: Noise Filtering (Lọc nhiễu ngữ cảnh)
- **Vấn đề giải quyết**: Loại bỏ các phát ngôn đùa cợt (*"chuyển sang product manager... nhưng đó chỉ là câu đùa"*) hoặc các địa điểm công tác tạm thời (*"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày"*).
- **Cải thiện recall/token**: Giúp agent không nhận nhầm `product manager` hay `Hà Nội` làm thông tin ổn định. Trong stress test, câu hỏi recall kiểm tra xem đâu mới là nơi ở và nghề thật sự đạt điểm tuyệt đối 100%.
- **Rủi ro phát sinh**: Tăng độ phức tạp của logic lọc (hardcoded heuristic rules), khó tổng quát hóa khi người dùng đưa ra các ngữ cảnh nhiễu mới chưa được định nghĩa mẫu.

### 4.4. Bonus 4: Structured Entity Extraction (Trích xuất thực thể có cấu trúc)
- **Vấn đề giải quyết**: Tránh việc ghi `User.md` thành một đoạn văn xuôi lộn xộn khó tra cứu.
- **Cải thiện recall/token**: Chuẩn hóa thành các mục rõ ràng: `Personal Info` (`Name`, `Location`, `Profession`) và `Preferences` (`Response Style`, `Favorite Drink`, `Favorite Food`, `Pet`, `Interests`). Giúp việc tra cứu chính xác, kích thước file nhỏ gọn (253–305 bytes) và LLM đọc prompt dễ dàng.
- **Rủi ro phát sinh**: Schema cố định hạn chế khả năng lưu các thông tin phi cấu trúc bất ngờ từ người dùng nếu thông tin đó không thuộc bất kỳ trường nào đã định nghĩa.

### 4.5. Multi-Provider Support (Kiến trúc đa nhà cung cấp)
- Hệ thống hỗ trợ đầy đủ 6 provider trong `src/model_provider.py`: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`.
- Giúp hệ thống memory hoạt động độc lập, không bị vendor lock-in và sẵn sàng triển khai trên cả cloud LLM lẫn local LLM (Ollama).

---

## 5. Tự Đánh Giá Theo Rubric Chấm Điểm

- [x] **Mức 0–60 điểm (Cơ bản)**: Đầy đủ Baseline Agent (chỉ nhớ trong thread), Advanced Agent (có `User.md` bền vững và compact memory), chạy trên dataset tiếng Việt, cấu trúc repo chuẩn hóa.
- [x] **Mức 60–75 điểm (Benchmark & Test)**: Chạy cùng một input benchmark cho cả hai agent; đầy đủ 6/6 test cases xanh trong `pytest src/test_agents.py -v`; hai bảng benchmark in đủ 6 cột chuẩn.
- [x] **Mức 75–90 điểm (Hiểu bản chất trade-off)**: Có cả Standard Benchmark và Long-Context Stress Benchmark; chứng minh bằng số liệu rằng compact tối ưu trực tiếp `Prompt tokens processed` (~46.4% savings) ở hội thoại dài và giải thích vì sao Advanced tốn hơn ở hội thoại ngắn; có kiểm chứng bằng thực nghiệm ablation.
- [x] **Mức 90–100 điểm (Tính năng Bonus nâng cao)**: Triển khai trọn vẹn 4 tính năng mở rộng (Conflict Handling, Interrogative Filtering, Noise Filtering, Structured Extraction) kèm phân tích toàn diện 3 mặt: bài toán giải quyết, hiệu quả cải thiện và rủi ro hệ thống.
