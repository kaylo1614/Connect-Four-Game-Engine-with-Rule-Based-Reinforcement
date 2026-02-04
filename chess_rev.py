"""
四子棋 AI 遊戲系統 - 數據驅動與啟發式邏輯版本
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pygame

# ==========================================
# 1. 數據初始化與載入
# ==========================================
Mset = [] # 儲存歷史盤面數據的清單
Wset = [] # 儲存對應盤面勝負結果的清單 (1: 電腦勝, -1: 玩家勝, 0: 平手)
train_time = 0 

# 從本地端載入預訓練好的 NumPy 二進位檔案
# 這些檔案包含了超過 16 萬筆的對戰狀態
try:
    Mset_ = np.load(r'Mset161113.npy')
    Wset_ = np.load(r'Wset161113.npy')

    for arr in Mset_:
        Mset.append(arr)
    for num in Wset_:
        Wset.append(num)
    
    train_time = len(Mset)
    print(f"成功載入數據，總筆數: {train_time}")
except FileNotFoundError:
    print("找不到數據檔，請確認 .npy 檔案路徑正確。")

# ==========================================
# 2. 核心邏輯演算法
# ==========================================

def next(N, row, col, x, y, val):
    """
    遞迴函式：計算在特定方向 (x, y) 上有多少個連續的相同棋子
    N: 盤面矩陣, row/col: 當前座標, x/y: 移動向量, val: 目前連續計數
    """
    if ((N[row][col] == N[row+x][col+y]) and (val < 4)):
        val += 1
    else:
        return 1
    return 1 + next(N, row+x, col+y, x, y, val)

def Check_win(N, row, col):
    """
    勝負判定演算法：檢查落子後是否達成四子連線
    """
    ans = 0
    # 建立一個邊界緩衝矩陣 A，避免掃描時發生 index out of range
    A = np.zeros((8, 9))
    for ii in range(0, 6):
        for jj in range(0, 7):
            if (N[ii][jj] == 0):
                pass
            elif ((N[ii][jj] % 2) == 0): # 偶數步為玩家（或後手）
                A[ii+1][jj+1] = -1
            else:                        # 奇數步為電腦（或先手）
                A[ii+1][jj+1] = 1

    # 1. 垂直檢查
    if(row >= 3):
        if (abs(np.sum(A[row-2:row+2, col+1])) == 4):
            ans = 1
            
    # 2. 右斜對角線檢查 (1, 1) 與 (-1, -1)
    count = next(A, row+1, col+1, 1, 1, 1) + next(A, row+1, col+1, -1, -1, 1) - 1
    if (count >= 4): ans = 1
    
    # 3. 左斜對角線檢查 (1, -1) 與 (-1, 1)
    count = next(A, row+1, col+1, 1, -1, 1) + next(A, row+1, col+1, -1, +1, 1) - 1
    if (count >= 4): ans = 1
    
    # 4. 水平檢查 (0, 1) 與 (0, -1)
    count = next(A, row+1, col+1, 0, 1, 1) + next(A, row+1, col+1, 0, -1, 1) - 1
    if (count >= 4): ans = 1
        
    return ans

def True_false(N, last):
    """
    啟發式攔截邏輯：預測下一步
    b: 必下點 (自己快贏了或對方快贏了)
    c: 禁下點 (下了之後會送對方贏)
    """
    a = [] # 儲存每一列目前可落子的最低行座標
    b = [] # 必下建議清單
    c = [] # 禁下警告清單
    A = N.copy()
    
    # 找到每一列最底部可落子的位置
    for jj in range(0, 7):
        k = 0
        for ii in range(0, 6):
            if(N[ii][jj] == 0):
                a.append(ii)
                k = 1
                break
        if(k == 0): a.append(-1)

    # 模擬：如果我方下這裡會贏嗎？
    for ii in range(0, 7):
        if (a[ii] >= 0):
            A[a[ii], ii] = last + 1
            if (Check_win(A, a[ii], ii) == 1):
                b.append(ii)
            A[a[ii], ii] = 0
            
    # 模擬：如果對方下這裡會贏嗎？(攔截)
    if (b == []):
        for ii in range(0, 7):
            if (a[ii] >= 0):
                A[a[ii], ii] = last + 2
                if (Check_win(A, a[ii], ii) == 1):
                    b.append(ii)
                A[a[ii], ii] = 0

    # 模擬：如果我下這裡，會不會導致對方下一手直接贏？(避免助攻)
    for ii in range(0, 7):
        if(a[ii] >= 0) and (a[ii] < 5):
            A[a[ii], ii] = last + 1
            A[a[ii]+1, ii] = last + 2
            if (Check_win(A, a[ii]+1, ii) == 1):
                c.append(ii)
            A[a[ii], ii] = 0
            A[a[ii]+1, ii] = 0
            
    if b == []: b.append(-1)
    if c == []: c.append(-1)
    return([b, c])

def Point_to_col(p_, TF):
    """
    決策引擎：將數據庫的分數 (p_) 結合攔截邏輯 (TF) 轉化為最終落子優先順序
    """
    p = [max(0, val) for val in p_] # 排除負分，只看獲勝期望
    
    count_ = p.count(0)
    max_ = max(p) if p else 1
    min_ = min(p) if p else 0
    
    # 計算權重放大係數 alpha
    if count_ == len(p_):
        alpha = 1
    else:
        alpha = 1 / (max_ - min_) if (max_ - min_) != 0 else 1
        
    grade = []
    final_result = []
    dont_put = []
    
    # 根據統計數據計算每一列的權重得分
    for ii in range(len(p)):
        if count_ == len(p_):
            grade.append(1)
        else:
            grade.append(alpha + 3 * (p[ii] - min_) / (max_ - min_) if (max_ - min_) != 0 else 1)
    
    # 優先處理必下點
    if TF[0][0] != -1:
        for ii in range(len(TF[0])):
            final_result.append(TF[0][ii])
            grade[TF[0][ii]] = 0 # 必下點已排入結果，將其得分歸零避開後續輪盤抽籤
    
    # 標記禁下點
    if TF[1][0] != -1:
        for ii in range(len(TF[1])):
            grade[TF[1][ii]] = 0
            dont_put.append(TF[1][ii])
    
    # 輪盤賭抽籤演算法 (Roulette Wheel Selection)：根據權重隨機挑選剩餘位置
    while sum(grade) != 0:
        pool = sum(grade)
        ball = pool * np.random.rand()
        count = 0
        path = 0
        for ii in range(len(grade)):
            if grade[ii] == 0:
                path += 1
            else:
                count += grade[ii]
                if ball <= count:
                    final_result.append(path)
                    grade[ii] = 0
                    break
                else:
                    path += 1
        
    # 最後補上禁下點（作為下策中的下策）
    for ii in range(len(dont_put)):
        if dont_put[ii] not in final_result:
            final_result.append(dont_put[ii])
    
    return final_result

# ==========================================
# 3. 數據比對函式 (檢索歷史數據)
# ==========================================

def Check_same_1(j, N):
    """比對當前盤面與歷史數據庫中第 j 筆數據是否一致"""
    result = np.nonzero(N)
    for _ in range(len(result[0])):
        r, c = result[0][_], result[1][_]
        if((Mset[j][r][c] - N[r][c]) % 2 != 0 or Mset[j][r][c] > len(result[0]) or Mset[j][r][c] == 0):
            return 0
    return 1

def Check_same_2(j, N):
    """比對當前盤面之「鏡像」與歷史數據是否一致（對稱性優化）"""
    result = np.nonzero(N)
    for _ in range(len(result[0])):
        r, c = result[0][_], result[1][_]
        if((Mset[j][r][6-c] - N[r][c]) % 2 != 0 or Mset[j][r][6-c] > len(result[0]) or Mset[j][r][6-c] == 0):
            return 0
    return 1

def Compute_point_2(N, ii):
    """
    統計核心：搜尋歷史數據中所有符合當前盤面的狀態，並計算各列勝率權重
    """
    result = np.zeros((7, 2)) # [0]: 正面(電腦贏)次數, [1]: 負面(玩家贏)次數
    for j in range(len(Mset)):
        if(Check_same_1(j, N)):
            next_step = np.where(Mset[j] == ii + 1)
            if(len(next_step[0])) > 0:
                if(Wset[j] > 0): result[next_step[1][0]][0] += 1
                else: result[next_step[1][0]][1] += 1

        if(Check_same_2(j, N)):
            next_step = np.where(Mset[j] == ii + 1)
            if(len(next_step[0])) > 0:
                if(Wset[j] > 0): result[6 - next_step[1][0]][0] += 1
                else: result[6 - next_step[1][0]][1] += 1

    final = np.zeros(7)
    for i in range(7):
        final[i] = result[i][0] - result[i][1] # 計算淨勝值
    if(ii % 2): final = -final # 根據先後手調整正負號
    
    return final

def Visual(N):
    """文字化輸出盤面，用於控制台除錯"""
    NN = np.char.chararray((6,7), unicode=True)
    for u in range(0,7):
        for q in range(0,6):
            if N[q][u] == 0: NN[5-q][u] = '*'
            elif N[q][u] % 2 == 1: NN[5-q][u] = 'o' # 先手標記
            elif N[q][u] % 2 == 0: NN[5-q][u] = 'x' # 後手標記
    return NN

# ==========================================
# 4. Pygame 遊戲主迴圈
# ==========================================

pygame.init()   
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # 全螢幕模式
pygame.display.set_caption('四子棋遊戲 - AI 混合決策版本') 
screen.fill((242, 242, 242))

# --- 介面文字設定 ---
font = pygame.font.SysFont("Times New Roman", 80)

# ==========================================
# 5. 難易度選擇階段 (透過數據量控制 AI 強度)
# ==========================================
def select_difficulty():
    global Mset, Wset
    # (此處省略部分重複的文字渲染程式碼，保留核心邏輯)
    # Easy: 5萬筆數據 / Medium: 10萬筆 / Hard: 全數據
    pass # 實際執行時會根據座標點擊進行 Mset 切片

# ==========================================
# 6. 主遊戲邏輯
# ==========================================
# ... (略過初始化動畫座標與載入圖片)

running = True
while running:
    # 初始化單局遊戲狀態
    N = np.zeros((6,7))
    # ... (遊戲回合切換邏輯)
    
    # 電腦回合實作示例
    # 1. 計算數據點數 (Compute_point)
    # 2. 啟發式攔截檢查 (True_false)
    # 3. 獲取優先順序名單 (Point_to_col)
    # 4. 執行落子動畫與矩陣更新
    
    # 事件處理：按下按鈕關閉程式
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
