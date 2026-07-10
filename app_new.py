import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import google.generativeai as genai

# ============================================================
# 1. 페이지 기본 설정
# ============================================================
st.set_page_config(page_title="스마트 팩토리 AI 품질 관리 시스템", layout="wide", initial_sidebar_state="expanded")

st.title("🏭 Smart Factory AI Quality Control Dashboard")
st.markdown("##### YOLO 객체 탐지 기반 양품/불량 판정 + 결함 유형 분석 시스템")
st.markdown("---")

# ============================================================
# 2. 세션 상태 초기화 (API Key 새로고침 증발 방지)
# ============================================================
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

# ============================================================
# 3. 사이드바 - 설정 패널
# ============================================================
st.sidebar.header("⚙️ 대시보드 설정")

st.sidebar.subheader("🔑 API 설정")
input_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
if input_key != st.session_state.api_key:
    st.session_state.api_key = input_key

st.sidebar.subheader("📁 모델 경로 설정")
model_path = st.sidebar.text_input("YOLO 탐지 모델 경로", value="model/best.pt")

st.sidebar.subheader("🎚️ 파라미터 조정")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 1.0, 0.25, 0.01)

st.sidebar.subheader("🏷️ 정상(양품) 클래스명 설정")
good_keyword = st.sidebar.text_input(
    "정상으로 간주할 클래스명 키워드 (소문자 기준, 쉼표로 구분 가능)",
    value="good",
    help="탐지된 클래스명에 이 키워드가 하나라도 포함되면 정상(양품)으로 판단하고, 그 외 클래스가 탐지되면 불량으로 판단합니다.",
)
good_keywords = [k.strip().lower() for k in good_keyword.split(",") if k.strip()]

use_ai_report = st.sidebar.checkbox("🤖 Gemini AI 리포트 생성", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("MVTec AD 기반 산업 부품 품질 검사 데모")


# ============================================================
# 4. 모델 로드
# ============================================================
@st.cache_resource
def load_model(path):
    return YOLO(path)


# ============================================================
# 5. Gemini 해석 함수
# ============================================================
def analyze_with_gemini(is_defect, defect_types, api_key):
    if not api_key:
        return "⚠️ 사이드바에 Gemini API Key를 입력하시면 AI 분석 리포트를 볼 수 있습니다."

    try:
        genai.configure(api_key=api_key)

        if is_defect:
            defect_summary = ", ".join(defect_types)
            prompt = f"""
당신은 스마트 팩토리 제조 공정 및 품질 관리(QC) 최고 전문가입니다.
비전 검사(YOLO) 모델이 제품을 스캔한 결과, 다음과 같은 결함 유형이 탐지되어 '불량'으로 판정되었습니다:
[{defect_summary}]

다음 3가지 요소를 포함해 한국어로 작성해 주세요:
1. 결함 유형별 의미: 각 결함이 제품 품질에 어떤 영향을 미치는지
2. 원인 추정: 해당 결함이 발생했을 만한 공정상 잠재적 원인 (금형 마모, 공구 파손, 이물질 유입 등)
3. 현장 조치 가이드라인: 라인 중단 여부, 동일 로트(Lot) 전수 검사 필요성 등 즉시 취할 행동 지침
"""
        else:
            prompt = """
당신은 스마트 팩토리 제조 공정 및 품질 관리(QC) 최고 전문가입니다.
비전 검사(YOLO) 모델이 제품을 스캔한 결과, 별도의 결함이 탐지되지 않아 '양품'으로 판정되었습니다.

다음 내용을 한국어로 간단히 작성해 주세요:
1. 현재 판정 결과에 대한 요약
2. 양품 판정이라도 주기적으로 점검해야 할 잠재적 리스크 포인트 (있다면)
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini 분석 중 오류가 발생했습니다: {e}"


def is_good_class(class_name, good_keywords):
    name_lower = class_name.lower()
    return any(k in name_lower for k in good_keywords)


# ============================================================
# 6. 모델 로드 (실패 시 중단)
# ============================================================
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"🚨 모델 로드 실패: '{model_path}' 경로에 학습된 .pt 파일이 있는지 확인해 주세요. (에러: {e})")
    st.stop()

# ============================================================
# 7. 이미지 업로드
# ============================================================
MAX_IMAGES = 5
uploaded_files = st.file_uploader(
    "품질 검사를 진행할 제품 이미지를 업로드하세요 (최대 5장)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.warning(f"최대 {MAX_IMAGES}장까지만 업로드 가능합니다. 상위 {MAX_IMAGES}장만 처리합니다.")
        uploaded_files = uploaded_files[:MAX_IMAGES]

    run_analysis = st.sidebar.button("🏭 전수 검사 시작", use_container_width=True)

    tab1, tab2 = st.tabs(["🔍 실시간 AI 품질 검사", "📊 통계 요약"])

    # --------------------------------------------------------
    # 탭 1: 실시간 검사
    # --------------------------------------------------------
    with tab1:
        defect_count = 0
        good_count = 0

        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### 📦 검사 대상 #{idx + 1} : `{uploaded_file.name}`")
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 원본 이미지")
                st.image(image, use_container_width=True)

            if run_analysis:
                with st.spinner("AI가 제품 표면을 검사 중입니다..."):
                    results = model.predict(source=image, conf=conf_threshold)
                    r = results[0]
                    im_array = r.plot(line_width=2)
                    im_rgb = im_array[..., ::-1]
                    res_image = Image.fromarray(im_rgb)

                    with col2:
                        st.subheader("⚡ AI 탐지 결과")
                        st.image(res_image, use_container_width=True)

                    # 탐지 데이터 정리
                    det_data = []
                    defect_types = set()
                    is_defect = False

                    if len(r.boxes) > 0:
                        for i, box in enumerate(r.boxes):
                            class_id = int(box.cls.item())
                            class_name = model.names[class_id]
                            conf = float(box.conf.item())
                            xyxy = box.xyxy.cpu().numpy().ravel()

                            if not is_good_class(class_name, good_keywords):
                                is_defect = True
                                defect_types.add(class_name)

                            det_data.append(
                                {
                                    "ID": i,
                                    "클래스명": class_name,
                                    "정확도(Conf)": f"{conf:.4f}",
                                    "BBox": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]",
                                }
                            )

                    # 최종 판정 배너
                    if len(det_data) == 0:
                        st.info("탐지된 객체가 없습니다. (Confidence Threshold를 낮춰보세요)")
                    elif is_defect:
                        defect_count += 1
                        defect_list_str = ", ".join(sorted(defect_types))
                        st.error(f"🚨 [불량 판정] {uploaded_file.name} → 결함 유형: **{defect_list_str}**")
                    else:
                        good_count += 1
                        st.success(f"✅ [양품 판정] {uploaded_file.name} → 결함 없음")

                    # 세부 결과 + AI 리포트
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.markdown("⚙️ **탐지 세부 결과**")
                        if det_data:
                            st.table(det_data)
                        else:
                            st.write("결과 없음")

                    with sub2:
                        st.markdown("🤖 **AI 분석 리포트 (by Gemini)**")
                        if not use_ai_report:
                            st.write("AI 리포트 옵션이 꺼져 있습니다.")
                        elif det_data:
                            st.info(
                                analyze_with_gemini(
                                    is_defect, sorted(defect_types), st.session_state.api_key
                                )
                            )
                        else:
                            st.write("탐지된 객체가 없어 리포트를 생략합니다.")
            else:
                with col2:
                    st.info("사이드바의 '전수 검사 시작' 버튼을 누르면 결과가 표시됩니다.")

            st.markdown("---")

    # --------------------------------------------------------
    # 탭 2: 통계 요약
    # --------------------------------------------------------
    with tab2:
        st.subheader("📊 생산 공정 분석 대시보드 (가상 요약)")
        st.markdown("본 탭은 검사 라인 전체의 누적 데이터를 바탕으로 한 통계 요약 화면입니다.")

        m1, m2, m3 = st.columns(3)
        m1.metric(label="오늘 총 검사 수량", value=f"{len(uploaded_files)} 건")
        m2.metric(label="설비 종합 효율 (OEE)", value="96.5 %", delta="+0.8%")
        m3.metric(label="공정 목표 수율", value="98.2 %", delta="-0.3% (주의)", delta_color="inverse")

        st.markdown("##### 📈 주요 부품별 결함 발생 추이 (샘플)")
        chart_data = np.random.randn(20, 3)
        st.line_chart(chart_data)

else:
    st.info("👋 좌측 사이드바에서 모델 경로 / Gemini API Key를 설정한 뒤, 상단에 검사할 이미지를 업로드하세요.")
