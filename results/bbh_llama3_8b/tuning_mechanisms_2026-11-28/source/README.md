# BBH follow-up · Why did Boolean Expressions lose accuracy?

目標是在同一模型、固定考卷上，分辨示範標籤比例、示範數量、更新步數與可訓練快取範圍如何改變準確率和預測方向。主要比較是同一批示範的 Full CT-KV 減 Standard ICL。這個實驗包不會更動 11/24 的實驗目錄。

## 事先指定的設計

| 部分 | 正式設定 | 新執行數 |
|---|---|---:|
| 1 標籤比例 | Boolean；10、20-shot × 示範 True 比例 20%、50%、80% × 3 批；每批 Standard ICL 一次、Full CT-KV 和 25% answer_attention CT-KV 各跑 optimizer seeds 0、1，均做 16 次更新 | 90 |
| 2 步數 | Boolean、Date；原 10-shot、原 3 批；Full CT-KV 分別訓練 4、8、32 步，optimizer seed 0 | 18 |
| 3 可訓練參數 | 兩任務；原 10、20-shot × 原 3 批；保留完整快取，各做 Key-only、Value-only，16 步，optimizer seed 0 | 24 |
| 4 效能拆解 | 隨訓練記錄，不新增設定；直接 ICL 記錄測試推論時間和峰值 | 0 |
| 5 隨機選 KV | 兩任務；原 20-shot × 原 3 批 × 25%、50%、75%；16 步，optimizer seed 0；每格一個隨機 mask | 18 |
| 新跑的配對對照 | 完整快取 16 步（兩任務 × 兩個 shot 數 × 三批）12 組；原 20-shot 的 answer_attention（兩任務 × 三批 × 三個預算）18 組 | 30 |
| **合計** | **每組 200 題；36,000 筆正式預測** | **180** |

主實驗固定 Boolean 同一批 200 題，106 題 True、94 題 False；Date 使用其原來的另外 200 題。Boolean 示例從另外 50 題抽，29 題 True、21 題 False。10-shot 的三種 True:False 是 2:8、5:5、8:2；20-shot 是 4:16、10:10、16:4。固定每一示範 seed 的標籤內隨機抽樣、標籤位置與順序，各方法共享完全相同的 prompt。每個比例的 20-shot 前十個示範與配對 10-shot 完全相同。其他階段使用 11/24 的原始示範 ID 和順序，便於核對。不同示範 seed 可抽到共同題目。

模型與先前一致，使用 Meta-Llama-3-8B-Instruct、BF16、凍結模型權重、直接 Q/A 提示、固定 greedy decoding 和嚴格比對。模型 revision 為 `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`。兩側 KV 同時訓練時，演算法和之前一樣使用 Adam、learning rate 0.001、token dropout 0.1、leave-one-out masking、microbatch 2、累積所有示範後做一次 optimizer 更新。每條 run 的 cosine learning-rate schedule 依該 run 的總步數設定。25% 等比率只計示範 token 位置，所有指示詞位置保留。

## Mac 與 Vast 操作

1. 在 Mac 下載 `BBH_Mechanisms_v1.zip`，從 Vast 的 Jupyter 檔案頁上傳到 `/workspace/`。建議單張 A100-SXM4-80GB、100 GB 磁碟及先前使用的 Python 3.10 / CUDA 12.1 環境。
2. 在 **Vast 的 Jupyter Terminal** 貼上下列整段。首次解壓才執行這段；新開 Terminal 或續跑請直接進入既有資料夾。若 ZIP 缺檔，先確認上傳位置。

```bash
cd /workspace &&
test ! -e bbh-mechanisms-v1 &&
python3 -m zipfile -t BBH_Mechanisms_v1.zip &&
python3 -m zipfile -e BBH_Mechanisms_v1.zip . &&
cd bbh-mechanisms-v1 &&
bash study.sh plan &&
bash study.sh setup &&
bash study.sh access &&
bash study.sh preflight &&
bash study.sh smoke
```

3. 預計正式工作要分段時，先跑 5 組真正的主實驗（10-shot、50% True、示範 seed 0），實測耗時。此步可略過。smoke 只跑少量題目，結果存於獨立目錄，不納入正式統計。access 會嘗試沿用原模型快取，需要 Hugging Face 授權時在隱藏輸入提示貼 read token。

```bash
cd /workspace/bbh-mechanisms-v1
bash study.sh start pilot
bash study.sh status pilot
```

4. 等 pilot 顯示 `5/5 complete` 且 `driver idle`，再啟動全部。若想先回答主要問題，將 `all` 換為 `main`。同一時間只啟動一個 driver。它會檢查並略過已完成的設定。

```bash
cd /workspace/bbh-mechanisms-v1
bash study.sh start all
bash study.sh status all
```

若要看即時日誌，在另一個 Jupyter Terminal 執行下列兩行。這裡的 Ctrl+C 只會退出查看日誌。

```bash
cd /workspace/bbh-mechanisms-v1
tail -f "$(ls -t results/driver-logs/*.log | head -n 1)"
```

| 分段指令 | 完整階段包含的組數 | 說明 |
|---|---:|---|
| `bash study.sh start main` | 90 | 含 5 組 pilot、訓練檢查點、效能量測 |
| `bash study.sh start steps` | 24 | 18 組新步數 + 6 組配對 10-shot 完整 16 步 |
| `bash study.sh start params` | 36 | 24 組 K/V + 12 組完整 16 步；與 steps 共用 6 組 |
| `bash study.sh start random` | 36 | 18 組 random + 18 組相同預算 attention |
| `bash study.sh start all` | 180 | 所有不重複設定；先完成的組會略過 |

暫停用 `bash study.sh stop`，等目前這一組寫完，以 `bash study.sh status` 確認 idle。續跑使用原本資料夾中的 `bash study.sh start all`。若 worker 失敗，driver 停下來並在日誌末尾列出失敗檔案，可於 `results/attempts/` 查看。處理問題後，以同一個 start 指令續跑，失敗的組會從頭開始。只有通過 200 題預測與驗證檢查並帶有 DONE 記號的正式結果會進分析。

5. 完成或提早暫停之後，在 Vast 內做分析與備份。

```bash
cd /workspace/bbh-mechanisms-v1 &&
bash study.sh status &&
bash study.sh analyze &&
bash study.sh archive
```

封存會列印 `/workspace/BBH_Mechanisms_XXXXXXXX.tar.gz` 與同名 `.sha256` 的實際檔名。在 Jupyter 下載這兩個檔案，在 **Mac Terminal** 執行（若有多個封存，以下會使用最新下載的 checksum）。

```bash
cd "$HOME/Downloads" &&
sumfile=$(ls -t BBH_Mechanisms_*.tar.gz.sha256 | head -n 1) &&
test -n "$sumfile" &&
shasum -a 256 -c "$sumfile"
```

看到檔名後方出現 `OK` 再結束租用。背景的 `start` 在 instance 存活時執行，在 Vast 停止或銷毀機器會中斷工作。archive 可備份部分結果。

## 怎麼判讀

| 假說 | 事先指定的比較與預期 |
|---|---|
| 標籤偏斜與掉分有關 | 固定 shot 與 seed，計算 Full 16 步減 ICL 的準確率與預測 True 比例。把 20% True、80% True 各與 50% True 比較，兩個偏斜方向分開看。若向偏多的類別移動、少數類別準確率下降，且偏斜時比平衡時掉更多分，支持此解釋。 |
| 增加示範能緩解掉分 | 同樣標籤比例、相同 seed，比較 20-shot 與 10-shot 各自的「調整後減調整前」。再比較這個差是否隨標籤比例而變。 |
| 可訓練快取太大可能有影響 | 在相同的完整快取上比較只訓練 K、只訓練 V 與雙側。兩個單側組可訓練參數數量相同，運算角色不同；若單側結果彼此不同，需考慮角色。25% attention 同時改變可見資訊與可訓練參數。它比完整 ICL 的最終差異混合了選擇與訓練。 |
| 某些任務可能只需較少更新 | 在同一條 16 步軌跡上看 0、2、4、8、16 步驗證曲線；獨立的 4、8、16、32 步訓練終點另做測試分數比較。每個 horizon 有自己的 cosine schedule。一條 16 步軌跡的第 4 步，與獨立 4 步訓練的終點含義不同。 |
| 選法是否勝過隨機 | 逐題、逐 seed 對齊 20-shot 的 random 與 answer_attention，在實際保留位元組相同的情況下比較準確率、時間與記憶體。 |

所有訓練組從示範池另留 10 題做驗證，Boolean 固定 5 True / 5 False。驗證題不進該組示範或正式測試集。在 0、2、4、8、16 步及實際終點（如 32）記錄每個正確答案 token 的平均負對數似然（越低越好）與 greedy 準確率。早停探索用驗證 NLL 選擇預先列好的 horizon，相同分數選較短者；比較的是被選到的設定與固定 16 步在測試集的差異。這項回顧分析本身沒有省掉 GPU 訓練時間。分段檢查與單次測試題評分使用不同資料。舊版已看過這 200 題的彙總，新機制假說仍應在獨立題集重驗。

每次留下 `results/runs/<id>/predictions.jsonl`、`validation.json`、`metrics.json` 及 `DONE.json`。分析輸出在 `results/analysis/`。先讀 `paired_summary.csv` 和 `paired_by_seed.csv`，再讀 `validation_checkpoints.csv` 及 `resource_profiles.csv`。Boolean 的預測 True 次數是模型回答 True 的總次數；另有真實 True 題與真實 False 題的各自準確率。Date 的選項答案使用它原本的精確比對評分。

配對比較先在同一示範批次內平均兩個 optimizer seed，再在三批示範之間計算平均與樣本標準誤（SD / √3）。顯示每批結果與方向；三批可共用示範，且考卷固定，該標準誤只描述這幾批示範間的變異。接近零的結果不能證明等效。一個歷史設定最多只有一個隨機挑選 seed，泛化時需保留這個條件。

時間記錄快取構建、選擇、forward、backward、optimizer（含 clipping 和 scheduler）、平均步時、adaptation 以及獨立的 validation/inference 時間。同步 GPU 計時本身也會帶來開銷，與未加量測的舊時數比較時應註明。記憶體保留 cache、gradients、Adam state 的 tensor bytes，以及 PyTorch live/peak allocated bytes。第一個 microbatch 額外計算 autograd 為 backward 保存的浮點張量儲存量（排除模型與 prefix storage）；此數值只代表該次樣本。不同時間點的量測和峰值不可相加當作總顯存。詳細表格定義請見 ANALYSIS.md。
