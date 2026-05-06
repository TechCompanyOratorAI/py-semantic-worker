# Ghi chú: Nguồn gốc điểm số Semantic Worker

Tài liệu này tóm tắt rõ chuỗi `Input -> Feature -> Công thức/Model -> Output` trong code hiện tại, kèm đánh giá về tính xác thực.

## 1. Input (Đầu vào)

Hệ thống nhận 4 nhóm dữ liệu chính:

- Transcript theo segment:
  - `segmentText`
  - `segmentNumber`
  - `startTimestamp`, `endTimestamp`
- Thông tin chủ đề:
  - `topic_name`
  - `topic_description`
- Dữ liệu slide:
  - `slides[].extractedText`
  - Có cơ chế tách virtual pages nếu một bản ghi chứa nhiều trang (`[Trang N]` / `[Slide N]`).
- Audio bài nói (nếu có):
  - Dùng để chấm speech quality.

Nguồn code:

- `src/services/semantic_service.py`
- `src/services/speech_quality_service.py`

---

## 2. Feature (Đặc trưng)

### 2.1 Feature ngữ nghĩa (text)

- Embedding vector từ `SentenceTransformer`.
- Cosine similarity cho các cặp:
  - Segment vs Topic.
  - Segment vs từng Slide.

### 2.2 Feature âm thanh (audio)

Tính các chỉ số âm học:

- Tỉ lệ phát âm / im lặng:
  - `voiced_ratio`, `silence_ratio`
- Pitch:
  - `pitch_mean`, `pitch_std`, `pitch_variation`
- Energy:
  - `energy_mean`, `energy_std`, `volume_variation`
- Spectral + MFCC:
  - `spectral_centroid_mean`, `mfcc_features`
- Hesitation:
  - các mẫu `silence` / `filler`
  - `hesitation_count`, `total_hesitation_time`, `hesitation_rate`

Nguồn code:

- `src/services/semantic_service.py` (embedding, similarity)
- `src/services/speech_quality_service.py` (audio metrics)

---

## 3. Công thức / Model

## 3.1 Model embedding đang dùng

Khởi tạo bằng:

- `SentenceTransformer(settings.EMBEDDING_MODEL)`

Giá trị mặc định trong settings:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Lưu ý: trong comment có nhắc E5 prefix (`query:` / `passage:`), nhưng model runtime phụ thuộc biến môi trường `EMBEDDING_MODEL`.

Nguồn code:

- `src/services/semantic_service.py`
- `src/config/settings.py`

## 3.2 Relevance score (segment so với topic)

\[
relevance\_score = cosine(segment\_embedding, topic\_embedding)
\]

## 3.3 Semantic score (segment so với toàn bộ slides)

Quy trình:

1. Tính `raw_sim` giữa segment và từng slide.
2. Lấy:
   - `best_raw`: điểm cao nhất.
   - `mean_raw`: trung bình tất cả điểm.
3. Chuẩn hóa tương phản:

\[
calibrated = \frac{best\_raw - mean\_raw}{1 - mean\_raw}
\]

4. Chặn miền giá trị về `[0,1]`.

Ý nghĩa: nếu mọi slide đều giống nhau về mức liên quan (không có slide nổi trội), điểm sẽ thấp.

## 3.4 Alignment score (khớp tiến độ thuyết trình)

Thành phần:

- `temporal_score`: khớp theo tiến độ thời gian/position.
- `content_match_score`: cosine giữa segment và **slide kỳ vọng**, có baseline correction.

Baseline correction:

\[
content\_match\_score = \frac{raw\_content - 0.55}{1 - 0.55}
\]

Sau đó chặn `[0,1]`.

Trộn điểm:

\[
alignment\_score = 0.35 \times temporal\_score + 0.65 \times content\_match\_score
\]

## 3.5 Speech quality scores

### Fluency

\[
hesitation\_penalty = \min(1.0, hesitation\_count / 10)
\]

\[
silence\_penalty = \min(1.0, silence\_ratio / 0.5)
\]

\[
fluency = \max(0, 1 - hesitation\_penalty - 0.3 \times silence\_penalty) \times rhythm\_score
\]

### Clarity

\[
spectral\_clarity = \min(1.0, spectral\_centroid\_mean / 2000)
\]

\[
energy\_consistency = 1.0 - \min(1.0, volume\_variation)
\]

\[
clarity = 0.4 \times spectral\_clarity + 0.3 \times energy\_consistency + 0.3 \times voiced\_ratio
\]

### Confidence

\[
pitch\_confidence = 1.0 - \min(1.0, pitch\_variation)
\]

\[
energy\_level = \min(1.0, energy\_mean / 0.2)
\]

\[
speaking\_rate\_score = \min(1.0, speaking\_rate / 180)
\]

\[
confidence = 0.4 \times pitch\_confidence + 0.3 \times energy\_level + 0.3 \times speaking\_rate\_score
\]

### Speech overall

\[
speech\_overall = 0.4 \times fluency + 0.3 \times clarity + 0.3 \times confidence
\]

## 3.6 Overall score của bài thuyết trình

Từ các segment:

\[
avg\_semantic = mean(segment.semantic\_score)
\]

\[
avg\_alignment = mean(segment.alignment\_score)
\]

Khi chưa có audio:

\[
overall = 0.5 \times avg\_semantic + 0.5 \times avg\_alignment
\]

Khi có audio:

\[
overall = 0.7 \times overall + 0.3 \times speech\_overall
\]

Ghi chú trong code: `content_relevance` có tính và lưu, nhưng không đưa vào công thức `overall_score`.

---

## 4. Output (Đầu ra)

### 4.1 Output theo segment

Mỗi segment trả về:

- `relevance_score`
- `semantic_score`
- `alignment_score`
- `best_matching_slide`, `slide_similarities`
- `issues`, `suggestions`
- `speech_quality` theo segment (nếu có audio)

### 4.2 Output tổng thể

- `content_relevance`
- `semantic_similarity`
- `slide_alignment`
- `overall_score`
- (nếu có audio) `speech_fluency`, `speech_clarity`, `speech_confidence`, `speech_overall`

---

## 5. Công thức này “từ đâu ra”?

Theo code hiện tại:

- Phần embedding + cosine: theo cách làm chuẩn của semantic similarity.
- Phần trọng số, baseline, ngưỡng, penalty:
  - là heuristic do nhóm thiết kế trong mã nguồn.
  - ví dụ: `0.35/0.65`, `0.4/0.3/0.3`, baseline `0.55`.

Nghĩa là đây không phải “định luật cố định” đúng cho mọi hệ, mà là thiết kế chấm điểm của hệ thống này.

---

## 6. Tính xác thực: “nó có đúng không?”

Câu trả lời chuẩn về mặt kỹ thuật:

- Không thể nói đúng tuyệt đối.
- Có thể nói: điểm số **nhất quán và tái lập được** vì input + công thức cố định.
- Độ đúng thực tế phải chứng minh bằng kiểm định:
  - So với chấm tay của giám khảo trên tập dữ liệu chuẩn.
  - Đo tương quan (Spearman/Pearson) giữa điểm hệ thống và điểm người chấm.
  - Đo sai số (MAE/RMSE) nếu có nhãn số.
  - Kiểm tra stability: cùng input chạy lại cho cùng output.

Kết luận nên dùng khi bảo vệ:

- Hệ thống là công cụ hỗ trợ đánh giá khách quan, có thể audit theo từng bước.
- Không thay thế hoàn toàn đánh giá chuyên gia nếu chưa có benchmark đủ mạnh.

---

## 7. Câu trả lời ngắn gọn cho hội đồng (gợi ý)

"Điểm trong semantic worker được tạo theo pipeline cố định: input transcript/slide/audio -> trích xuất feature -> áp công thức trong code -> trả output. Phần semantic dùng embedding + cosine similarity; phần speech dùng các chỉ số âm học như nhịp nói, độ ngắt quãng, pitch, energy. Các trọng số là heuristic được định nghĩa rõ trong mã nguồn nên điểm tái lập được. Tuy nhiên độ đúng tuyệt đối phải được xác nhận bằng benchmark với chấm tay của giám khảo; vì vậy hệ thống này nên xem là công cụ hỗ trợ đánh giá khách quan, không thay thế hoàn toàn chuyên gia." 
