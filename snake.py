import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import active_contour


def getGaussianPE(src):
    """
    描述：計算負高斯勢能(Negative Gaussian Potential Energy, NGPE)
    輸入：單通道灰度圖src
    輸出：無符號的浮點型單通道，取值0.0 ~ 255.0
    """
    imblur = cv.GaussianBlur(src, ksize=(5, 5), sigmaX=3)
    dx = cv.Sobel(imblur, cv.CV_16S, 1, 0)  # X方向上取一階導數，16位有符號數，卷積核3x3
    dy = cv.Sobel(imblur, cv.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E


def getDiagCycleMat(alpha, beta, n):
    """
    計算5對角循環矩陣
    """
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta
    diag_mat_a = a * np.eye(n)
    diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    return diag_mat_a + diag_mat_b + diag_mat_c


def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """
    以參數方程的形式，獲取n個離散點圍成的圓形/橢圓形輪廓
    輸入：中心centre=（x0, y0）, 半軸長radius=(a, b)， 離散點數N
    輸出：由離散點座標(x, y)組成的2xN矩陣
    """
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])


def getRectContour(pt1=(0, 0), pt2=(50, 50)):
    """
    根據左上、右下兩個頂點來計算矩形初始輪廓座標
    由於Snake模型適用於光滑曲線，故這裏用不到該函數
    """
    pt1, pt2 = np.array(pt1), np.array(pt2)
    r1, c1, r2, c2 = pt1[0], pt1[1], pt2[0], pt2[1]
    a, b = r2 - r1, c2 - c1
    length = (a + b) * 2 + 1
    x = np.ones((length), np.float)
    x[:b] = r1
    x[b : a + b] = np.arange(r1, r2)
    x[a + b : a + b + b] = r2
    x[a + b + b :] = np.arange(r2, r1 - 1, -1)
    y = np.ones((length), np.float)
    y[:b] = np.arange(c1, c2)
    y[b : a + b] = c2
    y[a + b : a + b + b] = np.arange(c2, c1, -1)
    y[a + b + b :] = c1
    return np.array([x, y])


def snake(img, snake, alpha=0.5, beta=0.1, gamma=0.1, max_iter=2500, convergence=0.01):
    """
    根據Snake模型的隱式格式進行迭代
    輸入：彈力系數alpha，剛性係數beta，迭代步長gamma，最大迭代次數max_iter，收斂閾值convergence
    輸出：由收斂輪廓座標(x, y)組成的2xN矩陣， 歷次迭代誤差list
    """
    x, y, errs = snake[0].copy(), snake[1].copy(), []
    n = len(x)
    # 計算5對角循環矩陣A，及其相關逆陣
    A = getDiagCycleMat(alpha, beta, n)
    inv = np.linalg.inv(A + gamma * np.eye(n))
    # 初始化
    y_max, x_max = img.shape
    max_px_move = 1.0
    # 計算負高斯勢能矩陣，及其梯度
    E_ext = -getGaussianPE(img)
    fx = cv.Sobel(E_ext, cv.CV_16S, 1, 0)
    fy = cv.Sobel(E_ext, cv.CV_16S, 0, 1)
    T = np.max([abs(fx), abs(fy)])
    fx, fy = fx / T, fy / T
    for g in range(max_iter):
        x_pre, y_pre = x.copy(), y.copy()
        i, j = np.uint8(y), np.uint8(x)
        try:
            xn = inv @ (gamma * x + fx[i, j])
            yn = inv @ (gamma * y + fy[i, j])
        except Exception as e:
            print("索引超出範圍")
        # 判斷收斂
        x, y = xn, yn
        err = np.mean(0.5 * np.abs(x_pre - x) + 0.5 * np.abs(y_pre - y))
        errs.append(err)
        if err < convergence:
            print(f"Snake迭代{g}次後，趨於收斂。\t err = {err:.3f}")
            break
    return x, y, errs


def main():
    src = cv.imread("circle.jpg", 0)
    img = cv.GaussianBlur(src, (3, 3), 5)

    # 構造初始輪廓線
    init = getCircleContour((140, 95), (110, 80), N=200)
    # Snake Model
    x, y, errs = snake(img, snake=init, alpha=0.1, beta=1, gamma=0.1)

    plt.figure()  # 繪製輪廓圖
    plt.imshow(img, cmap="gray")
    plt.plot(init[0], init[1], "--r", lw=1)
    plt.plot(x, y, "g", lw=1)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.figure()  # 繪製收斂趨勢圖
    plt.plot(range(len(errs)), errs)
    plt.show()


if __name__ == "__main__":
    main()
