import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지", layout="wide")
st.title("YOLO 멀티 객체 탐지기")

# 1. 모델 로드 (캐싱하여 재로딩 방지)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 모델 경로 (깃헙 리포지토리 기준 상대 경로로 설정)
MODEL_PATH = "_myWork/model/best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다: {e}")
    st.stop()

# 2. 이미지 업로드 UI (accept_multiple_files=True 추가 및 최대 5장 제한 안내)
MAX_IMAGES = 5
uploaded_files = st.file_uploader(
    f"탐지할 이미지를 업로드하세요 (최대 {MAX_IMAGES}장)", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    # 5장 초과 업로드 시 제한
    if len(uploaded_files) > MAX_IMAGES:
        st.warning(f"최대 {MAX_IMAGES}장까지만 업로드 가능합니다. 상위 {MAX_IMAGES}장의 이미지 만 처리합니다.")
        uploaded_files = uploaded_files[:MAX_IMAGES]

    # 파라미터 조정 UI (사이드바)
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    run_detection = st.sidebar.button("모든 이미지 탐지 실행")

    # 각 이미지별 처리를 위한 루프
    for idx, uploaded_file in enumerate(uploaded_files):
        # 구분선 및 이미지별 타이틀
        st.markdown(f"## 🖼️ 이미지 #{idx + 1} : {uploaded_file.name}")
        
        # PIL을 사용하여 이미지 읽기
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)

        # 탐지 버튼을 눌렀을 때 실행
        if run_detection:
            with st.spinner(f"'{uploaded_file.name}' 이미지를 분석하고 있습니다..."):
                # 3. 모델 예측
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
        else:
            with col2:
                st.info("사이드바의 '모든 이미지 탐지 실행' 버튼을 누르면 결과가 표시됩니다.")
        
        # 이미지 간의 구분을 위한 시각적 절취선
        st.markdown("---")
