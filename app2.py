import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지", layout="wide")
st.title("YOLO 객체 탐지기")

# 1. 모델 로드 (캐싱하여 재로딩 방지)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 모델 경로 (깃헙 리포지토리 기준 상대 경로로 설정)
# 실제 배포 시 이 경로에 best.pt 파일이 존재해야 합니다.
MODEL_PATH = "model/best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다: {e}")
    st.stop()

# 2. 이미지 업로드 UI
uploaded_file = st.file_uploader("탐지할 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # PIL을 사용하여 이미지 읽기
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    # 파라미터 조정 UI
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

    if st.sidebar.button("탐지 실행"):
        with st.spinner("이미지를 분석하고 있습니다..."):
            # 3. 모델 예측 (파일 경로 대신 PIL 객체 직접 전달)
            results = model.predict(source=image, conf=conf_threshold)
            r = results[0]
            
            # 4. 바운딩 박스 그려진 이미지 추출 (BGR -> RGB 변환)
            im_array = r.plot(line_width=2) 
            im_rgb = im_array[..., ::-1] 
            res_image = Image.fromarray(im_rgb)

            with col2:
                st.subheader("탐지 결과 이미지")
                st.image(res_image, use_container_width=True)

            # 5. 결과 정보 출력 (데이터프레임 형태)
            st.write("---")
            st.subheader("탐지된 객체 정보")
            st.write(f"**총 탐지된 객체 수:** {len(r.boxes)}")

            if len(r.boxes) > 0:
                det_data = []
                for i, box in enumerate(r.boxes):
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    conf = float(box.conf.item())
                    xyxy = box.xyxy.cpu().numpy().ravel()
                    
                    det_data.append({
                        "ID": i,
                        "클래스명": class_name,
                        "정확도(Conf)": f"{conf:.4f}",
                        "BBox (x1, y1, x2, y2)": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                    })
                st.table(det_data)