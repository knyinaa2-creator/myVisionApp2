import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
# 스트림릿 서버 배포 안정성이 가장 높은 라이브러리로 변경
import google.generativeai as genai

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지 + AI 해석", layout="wide")
st.title("YOLO 멀티 객체 탐지기 & AI 분석기")

# [보안/개선] API Key를 세션 상태에 저장하여 화면 새로고침 시 증발 방지
if "api_key" not in st.session_state:
    # 1순위: Streamlit Secrets / 2순위: 시스템 환경변수 / 3순위: 빈 값
    st.session_state.api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

st.sidebar.header("🔑 API 설정")
input_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.api_key)

# 사용자가 직접 입력하면 세션 상태 업데이트
if input_key != st.session_state.api_key:
    st.session_state.api_key = input_key

# [개선] 안정적인 호환 라이브러리 기반의 Gemini 해석 함수
def analyze_with_gemini(detected_items, api_key):
    if not api_key:
        return "⚠️ 사이드바에 Gemini API Key를 입력하시거나 Streamlit Secrets 설정을 완료하시면 AI 상세 분석을 볼 수 있습니다."
    
    try:
        # API 키 설정
        genai.configure(api_key=api_key)
        
        # 탐지 데이터를 텍스트로 요약
        items_summary = ", ".join([f"{item['클래스명']}(정확도: {item['정확도(Conf)']})" for item in detected_items])
        
        # 프롬프트 작성
        prompt = f"""
        YOLO 객체 탐지 모델이 이미지에서 다음과 같은 물체들을 찾아냈습니다:
        [{items_summary}]
        
        이 탐지 결과를 바탕으로 상황을 종합적으로 분석하고 다음 3가지 요소를 포함해 한국어로 자연스럽게 설명해 주세요:
        1. 현재 어떤 상황이나 장소로 추정되는지
        2. 탐지된 객체들이 의미하는 바나 중요하게 살펴볼 점
        3. 필요한 경우 주의사항이나 권장 행동
        """
        
        # 서버 호환성이 가장 좋은 gemini-2.5-flash 모델 사용
        model = genai.GenerativeModel('gemini-3.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini 분석 중 오류가 발생했습니다: {e}"

# 1. 모델 로드 (캐싱하여 재로딩 방지)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 모델 경로 (깃헙 리포지토리 기준 상대 경로)
MODEL_PATH = "model/best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다. (경로 확인: {MODEL_PATH}) \n에러 내용: {e}")
    st.stop()

# 2. 이미지 업로드 UI
MAX_IMAGES = 5
uploaded_files = st.file_uploader(
    f"탐지할 이미지를 업로드하세요 (최대 {MAX_IMAGES}장)", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.warning(f"최대 {MAX_IMAGES}장까지만 업로드 가능합니다. 상위 {MAX_IMAGES}장의 이미지 만 처리합니다.")
        uploaded_files = uploaded_files[:MAX_IMAGES]

    # 파라미터 조정 UI (사이드바)
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    run_detection = st.sidebar.button("모든 이미지 탐지 실행")

    # 각 이미지별 처리를 위한 루프
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"## 🖼️ 이미지 #{idx + 1} : {uploaded_file.name}")
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)

        if run_detection:
            with st.spinner(f"'{uploaded_file.name}' 이미지를 분석하고 있습니다..."):
                results = model.predict(source=image, conf=conf_threshold)
                r = results[0]
                
                im_array = r.plot(line_width=2) 
                im_rgb = im_array[..., ::-1] 
                res_image = Image.fromarray(im_rgb)

                with col2:
                    st.subheader("탐지 결과 이미지")
                    st.image(res_image, use_container_width=True)

                st.subheader("탐지된 객체 정보")
                st.write(f"**총 탐지된 객체 수:** {len(r.boxes)}")

                det_data = []
                if len(r.boxes) > 0:
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
                
                # 6. Gemini 분석 결과 출력
                st.subheader("🤖 AI 결과 해석")
                if len(det_data) > 0:
                    with st.spinner("Gemini가 탐지 결과를 분석하고 있습니다..."):
                        ai_analysis = analyze_with_gemini(det_data, st.session_state.api_key)
                        st.info(ai_analysis)
                else:
                    st.write("탐지된 객체가 없어 AI 해석을 진행하지 않습니다.")
                    
        else:
            with col2:
                st.info("사이드바의 '모든 이미지 탐지 실행' 버튼을 누르면 결과가 표시됩니다.")
        
        st.markdown("---")
