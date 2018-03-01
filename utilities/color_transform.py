import numpy as np

# for color augmentation, computed with make_pca.py
U = np.array([[-0.56543481, 0.71983482, 0.40240142],
              [-0.5989477, -0.02304967, -0.80036049],
              [-0.56694071, -0.6935729, 0.44423429]] ,dtype=np.float32)
EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)


def augment_color(img, sigma=0.1, color_vec=None):
    '''randomly change colors based on Krizhevsky color augmentation (gaussian)
       img: RGB numpy image (HxWxC)'''
    if color_vec is None:
        if not sigma > 0.0:
            color_vec = np.zeros(3, dtype=np.float32)
        else:
            color_vec = np.random.normal(0.0, sigma, 3)
    
    alpha = color_vec.astype(np.float32) * EV
    noise = np.dot(U, alpha.T)
    return img + noise[np.newaxis, np.newaxis, :]