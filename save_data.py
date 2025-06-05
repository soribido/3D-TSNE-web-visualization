import os
import pickle


# ---------------------------------------------------
# prob_list_x : List[List[float]]
# gt : List[str]
# image_paths : List[str]
# ---------------------------------------------------

def save_data(prob_list_x, gt, image_paths):
    assert len(prob_list_x) == len(gt) == len(image_paths), \
        "All three lists must have equal length. prob_list_x, gt, image_paths"

    data = {
        "features": prob_list_x,   # List[List[float]]
        "labels": gt,              # List[str]
        "image_paths": image_paths # List[str] 
    }

    pickle_path = "./saved_features_with_abs_paths.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)

    print(f"pickle 파일 저장 완료: {pickle_path}")