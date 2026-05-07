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

---

## 8. Pipeline phân tích giọng nói chi tiết

```
Audio file → Preprocess → VAD → openSMILE + Librosa → Metrics → Hesitations → Scores → Issues/Suggestions
```

### Bước 1 — Tiền xử lý audio (`_preprocess_audio`)
- Convert sang WAV nếu là MP3/MP4 (dùng pydub)
- Resample về sample rate chuẩn (mặc định 16000Hz)
- Normalize âm lượng
- Cắt bỏ khoảng lặng đầu/cuối (librosa.effects.trim, top_db=20)

### Bước 2 — Voice Activity Detection (`_detect_voice_activity`)
- Frame 25ms, hop 10ms
- Tính RMS energy theo từng frame
- Ngưỡng động: percentile 30 của energy — dưới ngưỡng là im lặng

### Bước 3 — Trích xuất features (2 nguồn song song)

**Librosa** (`_calculate_speech_metrics`):

| Feature | Cách tính | Ý nghĩa |
|---|---|---|
| `pitch_mean / pitch_std` | librosa.yin (fmin=50, fmax=400Hz) | Cao độ giọng trung bình / dao động |
| `energy_mean / energy_std` | librosa.feature.rms | Cường độ âm thanh |
| `speaking_rate` | voiced_ratio × duration × 3 × 60 / duration | Ước tính âm tiết/phút (~3 âm tiết/giây) |
| `voiced_ratio / silence_ratio` | voiced_frames / total_frames | Tỉ lệ nói / im lặng |
| `spectral_centroid_mean` | librosa.feature.spectral_centroid | Độ sáng/rõ của giọng |
| `mfcc_features` (13 hệ số) | librosa.feature.mfcc | "Dấu vân tay" âm học của giọng |
| `speech_rhythm_score` | 1 - std(voiced_energy)/mean(voiced_energy) | Độ đều nhịp khi nói |
| `volume_variation` | energy_std / energy_mean | Độ dao động âm lượng |
| `pitch_variation` | pitch_std / pitch_mean | Độ dao động cao độ |

**openSMILE** (`_extract_opensmile_features`):
- Feature set: `eGeMAPSv02` — chuẩn học thuật cho phân tích cảm xúc và chất lượng giọng
- Feature level: `Functionals` — thống kê tổng hợp trên toàn file
- Kết quả được merge vào dict metrics chung

### Bước 4 — Phát hiện ấp úng (`_detect_hesitation_patterns`)

**Loại `silence`** — khoảng im lặng bất thường:
- Ngắt > 200ms được đánh dấu là ấp úng
- `confidence = min(1.0, duration / 2.0)` — dài hơn = chắc hơn

**Loại `filler`** — từ đệm ("ừm", "à", ...):
- Đoạn có tiếng, ngắn 0.1s–1.0s
- Tính `spectral_centroid` và `spectral_bandwidth`
- `complexity_score = (centroid/1000) × (bandwidth/1000)`
- complexity < 0.5 → nhiều khả năng là từ đệm vô nghĩa

### Bước 5 — Tính điểm chất lượng (`_calculate_quality_scores`)

**Fluency (trôi chảy) — trọng số 40%:**
```
hesitation_penalty = min(1.0, hesitation_count / 10)
silence_penalty    = min(1.0, silence_ratio / 0.5)
fluency = max(0, 1 - hesitation_penalty - 0.3 × silence_penalty) × rhythm_score
```

**Clarity (rõ ràng) — trọng số 30%:**
```
spectral_clarity    = min(1.0, spectral_centroid_mean / 2000)
energy_consistency  = 1.0 - min(1.0, volume_variation)
clarity = 0.4 × spectral_clarity + 0.3 × energy_consistency + 0.3 × voiced_ratio
```

**Confidence (tự tin) — trọng số 30%:**
```
pitch_confidence    = 1.0 - min(1.0, pitch_variation)
energy_level        = min(1.0, energy_mean / 0.2)
speaking_rate_score = min(1.0, speaking_rate / 180)
confidence = 0.4 × pitch_confidence + 0.3 × energy_level + 0.3 × speaking_rate_score
```

**Speech Overall:**
```
speech_overall = 0.4 × fluency + 0.3 × clarity + 0.3 × confidence
```

### Bước 6 — Sinh issues và suggestions (`_generate_issues_and_suggestions`)

| Điều kiện | Issue | Suggestion |
|---|---|---|
| fluency < 0.6 | Low speech fluency | Practice speaking more smoothly |
| hesitation_count > 5 | Frequent hesitations | Prepare content better |
| clarity < 0.6 | Low speech clarity | Speak more clearly |
| voiced_ratio < 0.7 | Too much silence | Reduce long pauses |
| confidence < 0.6 | Low confidence indicators | Speak with more energy |
| speaking_rate < 100 spm | Speaking too slowly | Increase speaking pace |
| speaking_rate > 200 spm | Speaking too quickly | Slow down |
| pitch_variation > 0.8 | Inconsistent pitch | Practice steady vocal tone |

---

## 9. Q&A bảo vệ đồ án — câu hỏi thường gặp

### "Tại sao dùng openSMILE? Có thể dùng thư viện khác không?"

openSMILE là bộ công cụ trích xuất đặc trưng âm học được chuẩn hóa trong nghiên cứu học thuật, đặc biệt bộ feature set eGeMAPSv02 được thiết kế cho phân tích chất lượng giọng nói và nhận dạng cảm xúc. Dùng song song với librosa — librosa tính các chỉ số đơn giản như pitch, energy; openSMILE bổ sung các feature prosodic phức tạp hơn. Có thể thay bằng librosa thuần túy nhưng sẽ mất một phần đặc trưng.

### "Các trọng số 0.4, 0.3, 0.3 lấy từ đâu?"

Đây là câu hỏi nguy hiểm nhất — phải trả lời thật:

> Các trọng số này là heuristic được nhóm thiết kế dựa trên phán đoán thực nghiệm: fluency được ưu tiên cao nhất (0.4) vì ấp úng là dấu hiệu rõ ràng nhất của người nói chưa tự tin, clarity và confidence chia đều 0.3 còn lại. Đây là điểm hạn chế — để có trọng số tối ưu cần thu thập dataset có nhãn và dùng regression để học trọng số từ dữ liệu thực. Đây là hướng cải tiến tiếp theo.

### "Hệ thống phát hiện ấp úng chính xác đến đâu?"

> Hệ thống phát hiện hai dạng: khoảng im lặng trên 200ms và đoạn âm ngắn có độ phức tạp phổ thấp (từ đệm như "ừm", "à"). Chưa có benchmark với nhãn tay. Hệ thống cho kết quả nhất quán — cùng file audio luôn cho cùng kết quả — nhưng tương quan với đánh giá của người nghe thực tế là hướng cần kiểm định thêm.

### "Speaking rate tính như thế nào? Có đúng không?"

> Speaking rate được ước tính gián tiếp: lấy tỉ lệ thời gian có giọng nói nhân với 3 âm tiết/giây — con số trung bình cho tiếng Việt và tiếng Anh — rồi quy ra âm tiết/phút. Đây là xấp xỉ. Cải tiến tiếp theo: đếm số từ thực trong transcript từ ASR chia cho thời lượng để có speaking rate chính xác hơn.

### "Tại sao speech quality chỉ chiếm 30% tổng điểm?"

> Vì speech quality chỉ được tính khi có file audio. Nội dung (semantic + alignment) là yếu tố cốt lõi; giọng nói là yếu tố bổ trợ. Ngoài ra độ tin cậy của speech analysis hiện tại vẫn thấp hơn semantic analysis nên giữ trọng số 30% là hợp lý.

### "Hệ thống có hỗ trợ tiếng Việt không?"

> Phần semantic dùng model `paraphrase-multilingual-MiniLM-L12-v2` được train trên hơn 50 ngôn ngữ bao gồm tiếng Việt. Phần speech quality hoạt động ở tầng tín hiệu âm thanh — pitch, energy, silence đều đo được bất kể ngôn ngữ.

### "Nếu làm lại, em sẽ cải tiến gì?"

Ba điểm chính:
1. Thay speaking rate ước tính bằng đếm từ thực từ transcript ASR.
2. Thu thập dataset có nhãn từ giám khảo thực tế để học trọng số bằng regression thay vì heuristic.
3. Bổ sung phát hiện từ đệm tiếng Việt cụ thể ("ừ", "à", "thì", "là") bằng cách kết hợp ASR transcript thay vì chỉ dựa vào acoustic features.

---

## 10. Các điểm kỹ thuật sâu — hội đồng có thể hỏi bất ngờ

### 10.1 Tại sao dùng prefix "query:" và "passage:"?

**Vấn đề:** Model E5 (`intfloat/multilingual-e5-small`) được train với cơ chế asymmetric: văn bản cần tìm kiếm (query) và văn bản được tìm (passage) phải dùng prefix riêng. Nếu không có prefix, độ chính xác giảm đáng kể.

**Cách áp dụng trong code** (`semantic_service.py:184`):
```python
prefix = 'query: ' if is_query else 'passage: '
prefixed_texts = [prefix + t for t in valid_texts]
```

- Transcript segment (người nói) → `"query: <text>"` — đây là thứ đang tìm kiếm
- Slide content / Topic description → `"passage: <text>"` — đây là tài liệu được tìm

**Trả lời hội đồng:**
> "E5 là asymmetric embedding model — query và passage được encode vào không gian vector khác nhau để phản ánh vai trò tìm kiếm và được tìm. Nếu bỏ prefix, model sẽ hoạt động như symmetric encoder và mất đi ưu điểm thiết kế của nó."

---

### 10.2 Tại sao cần contrastive normalization cho semantic score?

**Vấn đề:** E5 multilingual cho cosine similarity **0.55–0.75 ngay cả khi hai văn bản không liên quan** (cùng ngôn ngữ). Điều này làm điểm bị lạm phát — mọi segment đều có vẻ khớp với mọi slide.

**Công thức hiệu chỉnh** (`semantic_service.py:318`):
```
calibrated = (best_raw - mean_raw) / (1 - mean_raw)
```

**Ý nghĩa:**
- Nếu tất cả slide đều cho điểm tương đương (không có slide nổi trội) → calibrated ≈ 0
- Nếu một slide rõ ràng khớp nhất → calibrated cao
- Kết quả: điểm thực sự phản ánh "slide này khớp hơn các slide khác bao nhiêu", không phải điểm tuyệt đối

**Trả lời hội đồng:**
> "Multilingual E5 có baseline cao — hai văn bản không liên quan cùng ngôn ngữ vẫn cho cosine ~0.55. Nếu dùng điểm raw, mọi segment sẽ có điểm cao dù không khớp slide nào. Contrastive normalization chuyển bài toán sang: segment này khớp slide nào *tốt hơn* các slide còn lại? Đây là câu hỏi có ý nghĩa hơn."

---

### 10.3 Tại sao content_relevance KHÔNG được tính vào overall_score?

**Đây là quyết định thiết kế quan trọng** — có comment rõ trong code (`semantic_service.py:669`):

> "Embedding similarity cannot reliably judge whether the speaker actually covered the topic — that requires LLM understanding."

**Lý do kỹ thuật:**
- Cosine similarity đo sự tương đồng từ vựng/ngữ nghĩa bề mặt
- Người nói có thể dùng từ khác hoàn toàn nhưng vẫn đúng chủ đề (paraphrase)
- Embedding không hiểu được logic, luận điểm, hay depth of coverage
- Để đánh giá thực sự người nói có bám chủ đề không → cần LLM (GPT/Claude)

**`content_relevance` vẫn được tính và lưu** cho từng segment để tham khảo, nhưng không đưa vào `overall_score`.

**Trả lời hội đồng:**
> "Chúng em cố tình loại content_relevance khỏi overall_score vì embedding không đủ độ tin cậy để phán xét người nói có bám chủ đề không. Người nói có thể diễn đạt đúng chủ đề bằng cách hoàn toàn khác với topic description, và cosine similarity sẽ cho điểm thấp dù nội dung đúng. Để làm tốt phần này cần LLM với khả năng hiểu ngữ nghĩa sâu hơn."

---

### 10.4 Xử lý slide nhiều trang (virtual slide expansion)

**Vấn đề:** Một file PDF 31 trang có thể được lưu trong database dưới dạng 1 bản ghi Slide duy nhất, với toàn bộ text gộp lại. Nếu dùng 1 slide cho 31 trang, alignment score sẽ hoàn toàn sai.

**Giải pháp** (`semantic_service.py:423`): Tách virtual slides theo marker `[Trang N]` / `[Slide N]`:
```
1 Slide DB record (31 pages) → 31 virtual slide entries
```

Alignment sau đó được tính dựa trên 31 virtual slides, cho kết quả chính xác theo từng trang.

**Trả lời hội đồng:**
> "Node API khi xử lý PDF nhiều trang sẽ ghi marker [Trang N] vào extractedText. Semantic worker detect marker này và expand 1 record thành nhiều virtual slides tương ứng số trang thực. Điều này đảm bảo alignment được tính ở mức độ trang, không phải mức độ file."

---

### 10.5 Alignment score: tại sao content-match chiếm 65%?

**Công thức** (`semantic_service.py:355`):
```
alignment_score = 0.35 × temporal_score + 0.65 × content_match_score
```

**Lý do content-match được ưu tiên hơn temporal:**
- Temporal score chỉ phản ánh vị trí thời gian — người nói đúng thứ tự chưa?
- Content-match phản ánh người nói có nói đúng nội dung slide tương ứng không?
- Một người có thể đúng thứ tự nhưng nói nội dung hoàn toàn khác với slide → điểm thấp
- Content-match là chỉ báo thực chất hơn của "khớp với slide"

**Baseline correction 0.55** cho content-match:
```
content_match = (raw_cosine - 0.55) / (1 - 0.55)
```
Lý do: E5 cho 0.55 baseline ngay cả khi không liên quan — cần shift out baseline này.

---

### 10.6 Tối ưu: embedding được tính 1 lần, dùng nhiều chỗ

**Vấn đề tiềm năng:** Mỗi segment cần embedding cho 3 mục đích: relevance, semantic similarity, alignment. Nếu tính 3 lần → lãng phí 3× inference time.

**Giải pháp trong code** (`semantic_service.py:560`):
```python
# Generate segment embedding ONCE — reused by semantic similarity AND
# alignment (content-match term). Avoids redundant model inference.
segment_embedding = self._generate_embeddings([segment_text], is_query=True)
```

Embedding được tính 1 lần duy nhất rồi truyền vào cả `_analyze_semantic_similarity` và `_analyze_alignment`.

**Trả lời hội đồng nếu hỏi về performance:**
> "Chúng em tối ưu bằng cách tính embedding của mỗi segment một lần duy nhất rồi reuse cho tất cả phép tính. Với model transformer, mỗi lần inference có chi phí không nhỏ, đặc biệt khi số segment lớn."

---

### 10.7 Chuẩn hóa timestamp ms ↔ giây

**Vấn đề thực tế:** ASR worker lưu timestamp dạng milliseconds (ví dụ `5200`), nhưng librosa tính thời gian dạng giây (`5.2`). Nếu so sánh trực tiếp sẽ sai hoàn toàn khi map hesitation patterns vào segment.

**Heuristic** (`semantic_service.py:546`):
```python
def _to_seconds(ts):
    return ts / 1000.0 if ts is not None and ts > 1000 else (ts or 0.0)
```

Nếu giá trị > 1000 → coi là milliseconds, chia 1000. Logic đơn giản nhưng hiệu quả vì timestamp giây hiếm khi vượt 1000s (~16 phút).

---

### 10.8 Điểm nào được lưu vào database, điểm nào không?

| Điểm | Lưu DB? | Ghi chú |
|---|---|---|
| `content_relevance` | Có (per segment) | Không vào overall_score |
| `semantic_similarity` | Có | 50% overall |
| `slide_alignment` | Có | 50% overall |
| `speech_fluency/clarity/confidence` | Có (nếu có audio) | Qua webhook → node-api |
| `speech_overall` | Có | 30% khi có audio |
| `overall_score` | Có | Điểm tổng bài |
| `hesitation_patterns` | Có (per segment) | Timestamp, loại, confidence |
| `mfcc_features` (13 hệ số) | Có (metadata) | Raw acoustic fingerprint |

---

### 10.9 Câu hỏi bẫy: "README nói dùng PostgreSQL nhưng code dùng MySQL?"

**Sự thật trong code** (`database_service.py:40`):
```python
self.connection = mysql.connector.connect(...)
```

**Trả lời thành thật:**
> "README được viết từ giai đoạn đầu khi dự án dự kiến dùng PostgreSQL. Trong quá trình tích hợp với hệ thống OratorAI tổng thể, team đã chuyển sang MySQL để đồng bộ với các service khác. README chưa được cập nhật — đây là technical debt về documentation mà chúng em nhận thấy."
