# Connect Four AI Game (Pygame Implementation)

### 專案簡介 / Project Overview
本專案實作了一個具備 AI 對戰功能的四子棋遊戲。除了基礎的遊戲機制與 GUI 介面外，核心亮點在於結合了「數據驅動」與「啟發式規則」的混合式決策邏輯，解決了在大狀態空間下，單純依靠數據模型可能產生的盲點。

This project implements a Connect Four game with an AI opponent. The highlight lies in the hybrid decision-making logic combining data-driven insights and heuristic rules to address model limitations in expansive state spaces.

### 技術亮點 / Technical Highlights
* **Hybrid Decision Logic**: 在訓練樣本覆蓋率不足的邊界情況下，設計了 **Hard Defense Constraints**（強制攔截邏輯），確保系統優先處理致命路徑，不遺漏對手的必勝點。
* **Heuristic Search**: 透過對盤面進行向量位移掃描，精準判定勝負條件與潛在陷阱。
* **Data Robustness**: 展現了在實務開發中，如何透過規則導向 (Rule-based) 的設計來補足模型訓練的局限性。

### 核心演算法 / Core Algorithms
* `Check_win`: 多向性勝負判定演算法，利用座標偏移量實作盤面掃描。
* `True_false`: **核心防禦邏輯**。針對樣本不足的路徑進行例外處理，強制實施防禦性落子。
* `Point_to_col`: 基於 16 萬筆對戰數據權重分佈的決策模型。

### 檔案說明 / Data Files
* **data.zip**: 包含核心訓練參數 `Mset161113.npy` 與 `Wset161113.npy`。因單一檔案體積限制，請下載後解壓縮至專案根目錄下的 `data/` 資料夾中。

### 如何執行 / How to Run
1. **環境安裝**: `pip install pygame numpy matplotlib`
2. **數據準備**: 解壓縮 `data.zip` 確保模型參數路徑正確。
3. **執行主程式**: `python main.py`
