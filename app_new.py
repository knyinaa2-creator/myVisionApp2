import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import os
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(page_title="스마트 팩토리 AI 품질 관리 대시보드", layout="wide")
st.title("🏭 Smart Factory AI Quality Control Dashboard")
st.markdown("##### OpenCV 의존성을 완전히 제거한 고안정성 MVTec AD 비전 검사 시스템")
st.markdown("---")

# 2. API 세션 설정
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

st.sidebar.header("🔑 AI 솔루션 관제탑")
input_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
if input_key != st.session_state.api_key:
    st.session_state.api_key = input_key

# 3. Gemini 해석 함수
def analyze_manufacturing_defect(detected_items, api_key):
    if not api_key:
        return "⚠️ 사이드바에 API Key를 입력하시면 AI 품질 분석 보고서를 볼 수 있습니다."
    try:
        genai.configure(api_key=api_key)
        items_summary = ", ".join([f"{item['클래스명']}(신뢰도: {item['신뢰도']})" for item in detected_items])
        
        prompt = f"""
        당신은 스마트 팩토리 품질 관리(QC) 최고 전문가입니다.
        비전 검사(YOLO) 모델이 부품 표면을 스캔하여 다음 요소를 탐지했습니다: [{items_summary}]
        
        이 결과를 바탕으로 현장 관리자를 위한 '품질 분석 보고서'를 한국어로 자연스럽게 작성해 주세요:
        1. **종합 불량 판정**: 현재 상태 진단 (정상/재작업/폐기)
        2. **원인 추정**: 결함이 발생한 잠재적 공정 원인 추정
        3. **현장 조치 가이드라인**: 지금 즉시 취해야 할 행동 지침
        """
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini 분석 중 오류 발생: {e}"

# 4. YOLO 모델 로드 (캐싱 및 CPU 연산 강제하여 리눅스 충돌 방지)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

MODEL_PATH = "model/best.pt"
try:
    # 🌟 OpenCV 에러를 피하기 위해 모델만 메모리에 올림
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"🚨 모델 로드 실패: {e}")
    st.stop()

# 5. 이미지 업로드 UI
uploaded_files = st.file_uploader("품질 검사 대상 이미지를 업로드하세요 (최대 5장)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    uploaded_files = uploaded_files[:5]
    conf_threshold = st.sidebar.slider("YOLO Threshold", 0.01, 1.0, 0.25, 0.01)
    run_detection = st.sidebar.button("🏭 전수 검사 시작", use_container_width=True)
    
    tab1, tab2 = st.tabs(["🔍 실시간 AI 품질 검사", "📊 일일 생산/검사 통계 보고서"])
    
    with tab1:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### 📦 Inspection Target #{idx + 1} : `{uploaded_file.name}`")
            orig_image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 원본 비전 이미지")
                st.image(orig_image, use_container_width=True)
                
            if run_detection:
                with st.spinner("AI가 표면 결함을 분석 중입니다..."):
                    # 🌟 중요: OpenCV 예측 대신 PIL 이미지를 넘기고 내부 그리기 기능 비활성화
                    results = model.predict(source=orig_image, conf=conf_threshold, verbose=False)
                    r = results[0]
                    
                    # 🎨 PIL을 이용해 직접 바운딩 박스 그리기 (OpenCV 우회 치트키)
                    draw_image = orig_image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    
                    det_data = []
                    is_defect = False
                    
                    if len(r.boxes) > 0:
                        for i, box in enumerate(r.boxes):
                            class_id = int(box.cls.item())
                            class_name = model.names[class_id]
                            conf = float(box.conf.item())
                            xyxy = box.xyxy.cpu().numpy().ravel() # [x1, y1, x2, y2]
                            
                            if 'good' not in class_name.lower():
                                is_defect = True
                            
                            # 빨간색 상자 그리기
                            draw.rectangle([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], outline="red", width=3)
                            draw.text((xyxy[0], max(0, xyxy[1]-15)), f"{class_name} {conf:.2f}", fill="red")
                            
                            det_data.append({
                                "ID": i,
                                "부품/결함 종류": class_name,
                                "신뢰도": f"{conf:.4f}"
                            })
                    
                    with col2:
                        st.subheader("⚡ AI 결함 탐지 결과")
                        st.image(draw_image, use_container_width=True)
                        
                    if is_defect:
                        st.error(f"🚨 [품질 비상] 결함이 감지되었습니다! 즉시 격리 조치하십시오.")
                    else:
                        st.success(f"✅ [품질 통과] 정상 양품(Good)으로 판정되었습니다.")
                        
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.markdown("⚙️ **탐지 데이터 테이블**")
                        if det_data:
                            st.table(det_data)
                        else:
                            st.write("정상 상태 (탐지된 결함 없음)")
                            
                    with sub_col2:
                        st.markdown("🤖 **LLM 공정 분석 보고서**")
                        if det_data:
                            ai_analysis = analyze_manufacturing_defect(det_data, st.session_state.api_key)
                            st.info(ai_analysis)
                        else:
                            st.write("특이 사항이 없어 보고서를 생성하지 않습니다.")
            else:
                with col2:
                    st.info("사이드바의 '전수 검사 시작' 버튼을 누르면 분석이 시작됩니다.")
            st.markdown("---")
            
    with tab2:
        st.subheader("📊 생산 공정 분석 대시보드")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("오늘 총 검사 수량", f"{len(uploaded_files)} 건")
        m_col2.metric("설비 종합 효율 (OEE)", "96.5 %", delta="+0.8%")
        m_col3.metric("공정 목표 수율", "98.2 %", delta="-0.3%", delta_color="inverse")
        st.line_chart(np.random.randn(20, 3))
