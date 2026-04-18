BODY25_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]

BODY25_SIZE = len(BODY25_NAMES)
HAND21_SIZE = 21
FACE_SIZE = 51

BODY25_KINTREE = [
    [1, 8],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [8, 9],
    [9, 10],
    [10, 11],
    [8, 12],
    [12, 13],
    [13, 14],
]

SMPLX_BODY25_SUPPORTED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23]
