# main.py
import numpy as np
import os, sys
import cv2


# copy this def function from ECE5242 Project 1.ipynb without explanation in here
# gaussian_pdf, gmm_pdf, segment_mask, find_components, merge_boxes_simple, bbox_height, bbox_base_point, predict_train_one, run_folder_to_results


# ---------- Load model ----------
def load_model(npz_path="gmm_cone_bg_hsv.npz"):
    model = np.load(npz_path)
    w_c, mu_c, Sig_c = model["w_c"], model["mu_c"], model["Sig_c"]
    w_b, mu_b, Sig_b = model["w_b"], model["mu_b"], model["Sig_b"]
    a = float(model["a"])
    return w_c, mu_c, Sig_c, w_b, mu_b, Sig_b, a

# ---------- GMM density ----------
# Gaussian density g_k(x|μk,Σk) D dimention Gaussian probability density
# 像素x在第k个颜色簇里有多像
# eps=1e-6 in case non-invertible or determinant = 0
def gaussian_pdf(X, mu, Sigma, eps=1e-6):
    """
    calculate g(x|mu,Sigma) for each x in X
    X: (N,D)
    mu: μ
    Sigma: Σ
    """
    # npy file uint8（0~255）need astype(np.float64) incase overflow
    X = X.astype(np.float64)
    mu = mu.astype(np.float64)
    Sigma = Sigma.astype(np.float64)

    N, D = X.shape # Dimention = 3; H,S,V

    Sigma = Sigma + eps * np.eye(D) # in case not invertable
    
    invS = np.linalg.inv(Sigma) # inverse matrix 逆矩阵
    detS = np.linalg.det(Sigma) # determinant 行列式

    diff = X - mu                              # diff to mean; shape (N,D)
    
    # 计算二次型 (x-μ)^T invS (x-μ) for each sample; "v^T * A * v"
    quad = np.sum((diff @ invS) * diff, axis=1)  # (N,) how far to mean

    coef = 1.0 / ((2*np.pi) **(D/2) * (detS**0.5)) # 归化一常数coefficient
    return coef * np.exp(-0.5 * quad) # return vector in (N,)


def gmm_pdf(X, w, mu, Sigma):
    """
    Compute mixture density p(x)
    X: (N,D)
    w: (K,)
    mu: (K,D)
    Sigma: (K,D,D)
    return: p (N,)
    """

    # npy file uint8（0~255）need astype(np.float64) incase overflow
    X = X.astype(np.float64)
    K = w.shape[0] # (K,) how many gaussion part
    N = X.shape[0] # X(N,D), X.shape[0] is N; N pixel sample; X.shape[1] is D

    # create a array length of N
    # p[i] is i-th sample mixture density
    p = np.zeros(N, dtype=np.float64)
    for k in range(K): # for every gaussion part, cumulate p += w[k] * g
        p += w[k] * gaussian_pdf(X, mu[k], Sigma[k])  # (N,)
    return p

# ---------- Segmentation ----------
def segment_mask(img_bgr, w_c, mu_c, Sig_c, w_b, mu_b, Sig_b, alpha=20.0):
    """
    input:
        img_bgr: (H,W,3) OpenCV read BGR image
        cone GMM: w_c, mu_c, Sig_c
        bg GMM: w_b, mu_b, Sig_b
    output:
        mask: (H,W) 0/1 matrix（black/white image）
        1 represent cone pixel
        0 represent background pixel
    """

    # turn image from BGR to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # (H,W,3)

    # get size
    H, W, _ = img_hsv.shape

    # turn every pixel into sample x=(H,S,V)
    X = img_hsv.reshape(-1, 3)  # get (N,3); N=H*W

    # use cone GMM to calculate every pixel density: p(x|cone)
    p_cone = gmm_pdf(X, w_c, mu_c, Sig_c)   # (N,)

    # use bg GMM to calculate each pixel density: p(x|bg)
    p_bg = gmm_pdf(X, w_b, mu_b, Sig_b)    # (N,)


    # pixel segment: more density means more like cone/bg
    # (p_cone > p_bg) get True/False, turn into 1/0; white is cone
    # do N comparison at the same time in array; Vectorized operations
    # tuned p_cone > alpha * p_bg to count as cone
    mask_flat = (p_cone > alpha* p_bg).astype(np.uint8) # (N,) 0/1 ; !!! Tune alpha

    mask = mask_flat.reshape(H,W) # (N,) result reshape to original shape (H,W)

    return mask

# ---------- Components / boxes ----------
def find_components(mask01, min_area=200): # min pixel use to filter small piece noise
    """
    mask01: (H,W) 0/1
    return list of boxes: (x,y,w,h)
    """
    mask01 = (mask01 > 0).astype(np.uint8) # make sure it's unit8 (0/1)
    
    # usage cite from OpenCV documentation
    # cv2.connectedComponentsWithStats() return retval, labels, stats, centroids 返回值、标签、统计信息、质心
    # stats[i] = [x, y, w, h, area] : i-th connected components info
    # [xy左上角axis,width,height,area:pixel number]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    
    boxes = []  #store [(x1, y1, x2, y2, area),(x1, y1, x2, y2, area),...]
    for i in range(1, num):  # skip 0 the biggest black background
        x, y, w, h, area = stats[i] 
        if h > w and area >= min_area: # cone's h > w and filter noise
            boxes.append((x, y, x+w-1, y+h-1, area)) # turn (x, y, w, h) into (x1, y1, x2, y2)
    return boxes


def merge_boxes_simple(boxes, x_th=30):
    """
    Merge boxes that are close.
    boxes: [(x1,y1,x2,y2,area), ...] (x1,y1)(x2,y2)左上右下角坐标的框
    return: [(x1,y1,x2,y2), ...] merged
    """
    if len(boxes) == 0:
        return []

    merged = []  # # each element is [x1,y1,x2,y2,cx]
    
    for (x1,y1,x2,y2,area) in boxes:
        cx = 0.5*(x1+x2) # calculate center x 'cx' 

        placed = False  # False means not merged yet
        for m in merged: # 
            if abs(cx - m[4]) < x_th: # if |current cx - merged_center| < threshold, it's same cone
                # merge into this group
                m[0] = min(m[0], x1) # 左上角取更小的 x/y
                m[1] = min(m[1], y1)
                m[2] = max(m[2], x2) # 右下角取更大的 x/y
                m[3] = max(m[3], y2)
                
                m[4] = 0.5*(m[0]+m[2]) # update group center (average, simple)
                
                placed = True # count as merged already
                break # then stop finding

        if not placed: # create new cone group
            merged.append([x1,y1,x2,y2,cx])

    return [tuple(m[:4]) for m in merged] # m[:4] means only return first 4 elements and no need area

def bbox_height(b): 
    x1, y1, x2, y2 = b
    return (y2 - y1 + 1) # return y axis height

def bbox_base_point(b):
    x1, y1, x2, y2 = b
    right = int((x1 + x2) / 2)
    down  = int(y2)
    return down, right

# ---------- Predict one cone ----------
def predict_train_one(img_bgr, a, min_area=200, x_th=30):
    """
    training set: 1 real cone per image
    pick the tallest bbox to avoid reflection
    return (down,right,dist) or None
    """
    mask = segment_mask(img_bgr, w_c, mu_c, Sig_c, w_b, mu_b, Sig_b, alpha=20.0)
    boxes = find_components(mask, min_area=min_area) # (x1,y1,x2,y2,area)
    bboxes = merge_boxes_simple(boxes, x_th=x_th)  # (x1,y1,x2,y2)

    if len(bboxes) == 0:
        return None

    # pick tallest bbox
    b = max(bboxes, key=bbox_height)
    down, right = bbox_base_point(b)
    dist = a / float(bbox_height(b)) # pre dict distance model
    return down, right, dist


# ---------- Run folder ----------
def run_folder_to_results(folder, out_txt, a, min_area=200, x_th=30):
    """
    folder: image path
    out_txt: name of output text: "results.txt"
    a: coefficient from distance model (d = a / h_pix)
    min_area: filter small noise connected component
    x_th: threshold of merge connected component
    """

    # list all file end with png and sort
    names = [n for n in os.listdir(folder) if n.endswith(".png")]
    names.sort()

    # open out_txt and overwrite/regenerate
    with open(out_txt, "w") as f:
        for fn in names:
            # read each image
            # do whole process (segment mask -> connect components get boxes -> merge get bboxes
            #  -> calculate each bbox base point + distance)
            img = cv2.imread(os.path.join(folder, fn))
            pred = predict_train_one(img, a, min_area=min_area, x_th=x_th)

            if pred is None: # if no cone write -1 -1 -1
                f.write(f"ImageName:{fn}, Down: -1, Right: -1, Distance: -1\n")

            else:
                down, right, dist = pred
                # look like ImageName:test_1.png, Down: 300.02, Right: 200.05, Distance: 450.6
                f.write(f"ImageName:{fn}, Down: {down:.2f}, Right: {right:.2f}, Distance: {dist:.2f}\n")


# ---------- main ----------
if __name__ == "__main__":
    # usage: python main.py <input_folder> <output_txt>
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_folder> <output_txt>")
        sys.exit(1)

    in_folder = sys.argv[1] # input image directory like ECE5242Proj1-test
    out_txt = sys.argv[2] # output filename like results.txt

    # load trained model + distance coefficient a
    w_c, mu_c, Sig_c, w_b, mu_b, Sig_b, a = load_model("gmm_cone_bg_hsv.npz")

    # run inference and write results
    run_folder_to_results(in_folder, out_txt, a, min_area=200, x_th=30)
    print("Saved:", out_txt)