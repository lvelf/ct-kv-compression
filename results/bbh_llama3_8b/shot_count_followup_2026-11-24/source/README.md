# BBH · Boolean 和 Date 的 144 組實驗

在同一批測試題上，比較示範數、示範 KV 保留率與更新次數對準確率、記憶體和時間的影響。本次從空的結果目錄開始，包含與先前 pilot 重疊的設定也會重新產生預測。

| 條件 | 示範 KV 保留率 | KV 更新 | 組數 |
| --- | --- | --- | ---: |
| answer_attention | 25%、50%、75% | 各 0、16 次 | 108 |
| Standard ICL | 100% | 0 次 | 18 |
| Full CT-KV | 100% | 16 次 | 18 |

兩個任務為 Boolean Expressions 和 Date Understanding，各固定 200 道測試題。示範 seed 為 0、1、2。每個 seed 的 5-shot 是 10-shot 的前五個示範，20-shot 接續同一組前十個示範。每個任務 × shot × seed 有 8 個設定，總共 144 組，全部可行時產生 28,800 筆逐題預測，涵蓋 400 道不同的測試題。

answer_attention 依示範答案的注意力分數挑選示範 token 的 KV。指示段全部保留。0 次更新也執行挑選；16 次更新會在相同選擇上繼續做 CT-KV。100% 的兩列分別是直接使用完整示範的 Standard ICL，以及在完整快取上更新 16 次的 Full CT-KV。相同百分比在不同 shot 數下可以對應不同的實際 KV 快取大小。

使用 Meta-Llama-3-8B-Instruct（revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`），BF16、greedy decoding、strict exact match、optimizer seed 0。正式排程無時間上限。

## 啟動

把 BBH_Attention_144.zip 上傳到 Vast Jupyter 的 /workspace。開一個 Vast Jupyter Terminal，在原實驗相同配置（單張 A100 80 GB、Python 3.10、CUDA 12.1）執行。如果上一套 720 組排程仍在同一台 GPU 上執行，先到 /workspace/bbh-four-fresh 執行 bash study.sh pause，等正在執行的那一組結束和舊 driver 退出，再啟動本次實驗。

```bash
cd /workspace &&
test ! -e bbh-attention-144 &&
python3.10 -m zipfile -t BBH_Attention_144.zip &&
python3.10 -m zipfile -e BBH_Attention_144.zip /workspace &&
cd /workspace/bbh-attention-144 &&
bash study.sh init &&
bash study.sh plan
```

新排程應顯示 complete=0、pending=144，其餘狀態為 0。init 會建立本次專用的 results 目錄。重新開 Terminal 後可直接用 status 或 start 接續這次的進度。

```bash
cd /workspace/bbh-attention-144 &&
bash study.sh setup &&
bash study.sh access &&
bash study.sh precheck &&
bash study.sh start
```

setup 核對環境並執行快取小測試。access 可共用先前下載的相同版本模型。若提示授權，貼入已獲 Llama 3 存取權的 Hugging Face read token，輸入會隱藏。precheck 用實際 tokenizer 預檢上下文長度；start 在背景依序執行並印出 driver 的 log 路徑。

## 進度、分析和備份

```bash
cd /workspace/bbh-attention-144 &&
bash study.sh status
```

要暫停可執行 bash study.sh pause。當前那一組寫完後，背景排程會停止。需要續跑時執行 bash study.sh start。若需先備份部分進度，等排程退出後用 bash study.sh archive。

當 pending、incomplete、mismatch 都為 0，complete 加 infeasible 為 144，且 driver 已退出後，執行

```bash
cd /workspace/bbh-attention-144 &&
bash study.sh finish
```

從 Jupyter 的 /workspace 下載最後印出的 BBH_Attention_144_*.tar.gz 和配對的 .tar.gz.sha256。Mac 上可把兩個檔案放在 Downloads，再核對下載檔案

```bash
cd "$HOME/Downloads" &&
printf '貼上這次 .tar.gz.sha256 的檔名，再按 Enter：' &&
IFS= read -r checkfile &&
shasum -a 256 -c "./$checkfile"
```

結果表位於 results/analysis-shots-two/。

| 檔案 | 用途 |
| --- | --- |
| shot_effects.csv | 看每個任務及兩任務平均在 5、10、20-shot 的準確率、成對差值與跨示範 seed 的差值標準差。 |
| contrasts_per_task.csv、contrasts_macro.csv | 看同保留率下 16 次相對 0 次更新的效果，以及同更新次數下壓縮相對完整快取的差異。準確率差以百分點表示。 |
| absolute_per_task.csv、absolute_macro.csv | 看準確率、保留快取 MiB、峰值 GPU GiB、適應秒數和推論秒數。 |
| runs.csv | 核對各設定的狀態、題數與對應的原始輸出。 |

兩任務平均給每個任務相同權重。正式比較需要對應的任務、seed 和設定都有結果；未取得的分數與差值保留空白。本設計可以評估 attention 壓縮相對完整快取的差異。比較 attention 與隨機挑選的優劣，需要同保留率的隨機對照。
