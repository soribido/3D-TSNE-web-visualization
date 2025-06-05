# main.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
from urllib.parse import quote, unquote
import urllib
import uvicorn

app = FastAPI()

# 1) 템플릿 디렉터리 지정 (templates/index.html 사용)
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 2) pickle 파일 경로 지정
PICKLE_PATH = os.path.join(BASE_DIR, "saved_features_with_abs_paths.pickle")

# 3) TSNE 메타 정보를 담을 전역 변수
tsne_meta = None


@app.on_event("startup")
def compute_tsne_and_meta():
    """
    서버가 시작될 때 한 번만 호출되어 pickle을 로드하고 TSNE(3D)를 계산합니다.
    이후 tsne_meta 리스트를 만들어 API 응답에 사용합니다.
    """
    global tsne_meta

    # 1) Load pickle file
    if not os.path.exists(PICKLE_PATH):
        raise RuntimeError(f"Cannot find Pickle : {PICKLE_PATH}")

    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    features = np.array(data["features"])  # shape = (N, D)
    labels = data["labels"]                # List[str], len N
    image_paths = data["image_paths"]      # List[str], len N

    # 2) 3D TSNE 
    tsne = TSNE(
        n_components=3,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        init="random"
    )
    coords_3d = tsne.fit_transform(features)  # shape = (N, 3)

    # 3) generate tsne_meta list
    meta_list = []
    for i, (coord, label, abs_path) in enumerate(zip(coords_3d, labels, image_paths)):
        encoded = quote(abs_path, safe="")  
        img_url = f"/get_image?path={encoded}"
        meta_list.append({
            "idx": i,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "label": label,
            "img_url": img_url
        })

    tsne_meta = meta_list


# 4) root page: templates/index.html rendering
@app.get("/")
def read_index(request: Request):
    """
    index.html을 렌더링하여 반환
    """
    return templates.TemplateResponse("index.html", {"request": request})


# 5) TSNE data return API
@app.get("/api/tsne_data")
def get_tsne_data():
    """
    tsne_meta List({idx, x, y, z, label, img_url}) -> JSON return
    """
    if tsne_meta is None:
        raise HTTPException(status_code=500, detail="TSNE data is None")
    return JSONResponse(content=tsne_meta)


# 6) 절대경로 이미지를 반환하는 API
@app.get("/get_image")
def get_image(path: str):
    # 클라이언트가 보내온 path는 quote()된 절대경로임
    abs_path = unquote(path)  
    print(f"[get_image] 디코딩된 경로 → {repr(abs_path)}")

    # 보안 체크: 실제 운영 시에는 반드시 허용된 디렉터리 내인지 검사해야 함
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(abs_path, media_type="image/jpeg")


if __name__ == "__main__":
    # host와 port를 지정하여 uvicorn 실행
    uvicorn.run("main:app", host="0.0.0.0", port=9604, reload=False)
