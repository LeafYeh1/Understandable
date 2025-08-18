# Understandable

##  一、專案簡介（摘要）
本作品為一套整合深度學習與語音分析的情緒辨識系統，協助諮商師更即時且客觀地掌握使用者的情緒變化。因在傳統諮商過程多仰賴心理師的觀察與個案主觀陳述，易受語境與情緒掩飾影響，辨識準確度有限。
為提升實用性，我們設計了可部署於網頁的應用，提供即時分析功能。系統介面將以折線圖呈現情緒隨著時間變化、圓餅圖顯示情緒分佈，並附加文字建議以輔助心理師判斷。

此工具能於諮商過程中即時提供客觀佐證，亦可長期追蹤個案情緒狀態，提升診斷與治療成效。面對現代社會壓力日增，情緒難以表達的挑戰，此系統有望成為心理健康領域的有利輔助。

## 二、成果亮點
- **模型準確率**：98.82%（以 RAVDESS/CREAM_D/TESS 測試集評估）
- **網站準確率**：83.33% 
- **回傳時間**：約 200 ms/筆（含網頁請求與模型推論）
- **界面友善**：即時圖表 + 情緒建議文字輸出
- **實用方向**：原型已部署於 Flask Web，適合作為心理、學習輔助等工具演示

## 三、技術架構與功能拆解 
| 組件         | 技術內容與功能                                                   |
|--------------|------------------------------------------------------------------|
| 前端        | HTML + JavaScript 〈支援錄音及音檔上傳與圖表呈現〉                     |
| 後端        | Python (Flask) + CNN - BiGRU 模型載入 + LLM(Qwen2.5) + 情緒計算 + PostgreSQL  |
| 模型        | CNN-BiGRU 組合（輸入 MFCC → 分類情緒類別）                        |
| 資料格式    | 上傳支援 `.wav` 或 `.webm` 音檔                                     |
| 額外工具    | `requirements.txt`（依賴）、`.gitignore`                          |

## 四、範例輸入與預期輸出
### 1. 選擇登入平台的角色
<img width="616" height="350" alt="螢幕擷取畫面 2025-08-03 170033" src="https://github.com/user-attachments/assets/ac23f6f7-c228-4bdb-b00b-2d489b8f13b0" />

### 2. 使用者與諮商師的註冊介面

<img width="347" height="300" alt="螢幕擷取畫面 2025-08-03 172205" src="https://github.com/user-attachments/assets/47c5bf96-8d40-4798-91a5-9233c9bd7b99" />
<img width="323" height="300" alt="螢幕擷取畫面 2025-08-03 172227" src="https://github.com/user-attachments/assets/187bb34a-fd63-4b8a-870a-c980d60e15f4" />

### 3. 錄音/上傳檔案辨識介面

<img width="404" height="300" alt="螢幕擷取畫面 2025-08-03 172548" src="https://github.com/user-attachments/assets/cedefa55-7eb4-4e3d-b279-bfa9a747e5cc" />
<img width="368" height="300" alt="螢幕擷取畫面 2025-08-03 172623" src="https://github.com/user-attachments/assets/9348f5a3-c15f-4af8-9fb1-be3334655040" />

### 4. PDF 輸出
<img width="329" height="549" alt="螢幕擷取畫面 2025-08-16 154457" src="https://github.com/user-attachments/assets/5b8ecbeb-1c46-486b-9059-031506d878a8" />

### 5. 患者資訊
<img width="634" height="350" alt="螢幕擷取畫面 2025-08-03 173105" src="https://github.com/user-attachments/assets/10207232-6328-4025-804e-ad769aebd3ee" />
