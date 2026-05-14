import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="NHANES 다이어트 성공 시뮬레이터",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diet_success_rf_model.joblib")
LR_MODEL_PATH = os.path.join(BASE_DIR, "diet_success_lr_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "diet_success_scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "diet_success_features.joblib")
DATA_PATH = os.path.join(BASE_DIR, "processed_data/integrated_nhanes.csv")

# 스타일 설정
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton > button { width: 100%; border-radius: 12px; height: 3em; font-weight: 600; }
    .info-box, .tip-box { padding: 1.5rem; border-radius: 12px; display: flex; flex-direction: column; margin-bottom: 1rem; }
    .info-box { background-color: #f0f2f6; border-left: 5px solid #1f77b4; }
    .tip-box { background-color: #fff9e6; border-left: 5px solid #ffcc00; }
    .flex-container { display: flex; gap: 20px; width: 100%; margin-top: 20px; }
    .block-container { padding-top: 4rem !important; padding-bottom: 1rem !important; }
    
    @media (max-width: 768px) { 
        .flex-container { flex-direction: column; gap: 10px; }
        h1 { font-size: 1.3rem !important; margin-bottom: 0.5rem !important; }
        h3 { font-size: 0.9rem !important; }
        [data-testid="stNumberInput"] { transform: scale(0.95); transform-origin: left top; }
        [data-testid="stSlider"] { transform: scale(0.95); transform-origin: left top; }
        .block-container { padding-top: 4.5rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if 'user_data' not in st.session_state: st.session_state.user_data = None
if 'nickname' not in st.session_state: st.session_state.nickname = ""
if 'menu_index' not in st.session_state: st.session_state.menu_index = 0
if 'input_step' not in st.session_state: st.session_state.input_step = 1

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=['RIDAGEYR', 'BMXBMI', 'LBXSGL', 'LBXGH', 'BMXWAIST', 'BPXSY1', 'BPXDI1'])
    return pd.read_csv(DATA_PATH)

def get_group_averages(age, bmi):
    try:
        df = load_data()
        age_group = (age // 10) * 10
        mask = (df['RIDAGEYR'] >= age_group) & (df['RIDAGEYR'] < age_group + 10)
        f = df[mask]
        if len(f) < 5: f = df
        return {
            'LBXSGL': f['LBXSGL'].median() if 'LBXSGL' in f else 100,
            'LBXGH': f['LBXGH'].median() if 'LBXGH' in f else 5.5,
            'BMXWAIST': f['BMXWAIST'].median() if 'BMXWAIST' in f else 90,
            'BPXSY1': f['BPXSY1'].median() if 'BPXSY1' in f else 120,
            'BPXDI1': f['BPXDI1'].median() if 'BPXDI1' in f else 80,
            'LBXTC': 180, 'LBXTR': 130, 'LBDHDD': 50, 'LBDLDL': 110
        }
    except: return {'LBXSGL': 100, 'LBXGH': 5.5, 'BMXWAIST': 90, 'BPXSY1': 120, 'BPXDI1': 80, 'LBXTC': 180, 'LBXTR': 130, 'LBDHDD': 50, 'LBDLDL': 110}

@st.cache_resource
def load_assets():
    return joblib.load(MODEL_PATH), joblib.load(LR_MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURES_PATH)

# 사이드바
pages = ["🏠 홈", "🔍 내 상태 확인", "📈 성공률 예측 시뮬레이션"]
with st.sidebar:
    selected_page = st.radio("메뉴:", pages, index=st.session_state.menu_index)
    if selected_page != pages[st.session_state.menu_index]:
        st.session_state.menu_index = pages.index(selected_page); st.rerun()

menu = pages[st.session_state.menu_index]

if menu == "🏠 홈":
    st.title("🍎 NHANES 비만 성공률 시뮬레이터")
    if not st.session_state.nickname:
        nick = st.text_input("닉네임 입력:", placeholder="다이어터A")
        if st.button("시작하기"):
            if nick: st.session_state.nickname = nick; st.rerun()
    else:
        st.subheader(f"안녕하세요, {st.session_state.nickname}님!")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎯 내 상태 입력하러 가기", use_container_width=True): st.session_state.menu_index = 1; st.rerun()
        with c2:
            if st.button("📈 바로 시뮬레이션 하기", use_container_width=True): st.session_state.menu_index = 2; st.rerun()

    st.markdown("""
        <div class="flex-container">
            <div class="info-box">
                <h3 style='color: #1f77b4;'>📋 분석 모델 정보</h3>
                <ul>
                    <li><b>데이터</b>: NHANES 2013-2014 통합 데이터</li>
                    <li><b>알고리즘</b>: Hybrid AI (RF + LR)</li>
                    <li><b>설명가능성</b>: 어떤 지표가 성공에 기여했는지 AI가 분석</li>
                </ul>
            </div>
            <div class="tip-box">
                <h3 style='color: #d4a017;'>💡 활용 팁</h3>
                <ul>
                    <li>목표 체중을 입력하면 감량 난이도가 자동으로 계산됩니다.</li>
                    <li>수치를 모르면 AI가 체형 기반 평균값으로 자동 보완합니다.</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif menu == "🔍 내 상태 확인":
    st.title("🔍 내 상태 및 목표 설정")
    st.progress(st.session_state.input_step / 3)

    if st.session_state.input_step == 1:
        st.subheader("1단계: 기본 신체 정보")
        age = st.number_input("나이 (세) *필수:", 18, 80, value=st.session_state.get('age', 35))
        gender = st.selectbox("성별 *필수:", ["남성", "여성"], index=0 if st.session_state.get('gender_str') == "남성" else 1)
        height = st.number_input("키 (cm) *필수:", 120.0, 220.0, value=st.session_state.get('height', 170.0))
        weight = st.number_input("현재 체중 (kg) *필수:", 40.0, 200.0, value=st.session_state.get('weight', 75.0), step=1.0)
        if st.button("다음으로 ➡️", use_container_width=True):
            st.session_state.age, st.session_state.gender_str = age, gender
            st.session_state.height, st.session_state.weight = height, weight
            st.session_state.input_step = 2; st.rerun()

    elif st.session_state.input_step == 2:
        st.subheader("2단계: 건강 지표 (선택)")
        waist = st.number_input("허리 둘레 (cm):", 50.0, 150.0, value=st.session_state.get('waist', None), placeholder="모르면 비워두세요", step=1.0)
        glucose = st.number_input("공복 혈당 (mg/dL):", 50, 300, value=st.session_state.get('glucose', None), placeholder="모르면 비워두세요", step=1)
        hba1c = st.number_input("당화혈색소 (%):", 3.0, 15.0, value=st.session_state.get('hba1c', None), placeholder="모르면 비워두세요", step=0.1)
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("⬅️ 이전"): st.session_state.input_step = 1; st.rerun()
        with c2: 
            if st.button("다음으로 ➡️"):
                st.session_state.waist, st.session_state.glucose, st.session_state.hba1c = waist, glucose, hba1c
                st.session_state.input_step = 3; st.rerun()

    elif st.session_state.input_step == 3:
        st.subheader("3단계: 다이어트 목표")
        target_w = st.number_input("목표 체중 (kg) *필수:", 30.0, 200.0, value=st.session_state.weight - 5.0, step=1.0)
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("⬅️ 이전"): st.session_state.input_step = 2; st.rerun()
        with c2: 
            if st.button("✅ 설정 완료"):
                bmi = st.session_state.weight / ((st.session_state.height/100)**2)
                avgs = get_group_averages(st.session_state.age, bmi)
                st.session_state.user_data = {
                    'RIDAGEYR': st.session_state.age, 'RIAGENDR': 1 if st.session_state.gender_str == "남성" else 2,
                    'BMXBMI': bmi, 'BMXWAIST': st.session_state.waist if st.session_state.waist else avgs['BMXWAIST'],
                    'LBXSGL': st.session_state.glucose if st.session_state.glucose else avgs['LBXSGL'],
                    'LBXGH': st.session_state.hba1c if st.session_state.hba1c else avgs['LBXGH'],
                    'BPXSY1': avgs['BPXSY1'], 'BPXDI1': avgs['BPXDI1'], 'LBXTC': avgs['LBXTC'], 'LBXTR': avgs['LBXTR'],
                    'LBDHDD': avgs['LBDHDD'], 'LBDLDL': avgs['LBDLDL'], 'RXDCOUNT': 0, 'PAQ610': 0, 'PAQ655': 0,
                    'CURRENT_W': st.session_state.weight, 'TARGET_W': target_w
                }
                st.success("🎉 설정 완료! 시뮬레이션을 시작합니다."); st.session_state.menu_index = 2; st.session_state.input_step = 1; st.rerun()

elif menu == "📈 성공률 예측 시뮬레이션":
    if st.session_state.user_data is None:
        st.error("⚠️ 신체 데이터를 먼저 입력해주세요!"); st.button("내 상태 입력하러 가기", on_click=lambda: setattr(st.session_state, 'menu_index', 1))
    else:
        u = st.session_state.user_data
        st.title("📈 성공률 예측 시뮬레이션")
        st.subheader(f"🎯 목표: {u['CURRENT_W']}kg → {u['TARGET_W']}kg")
        
        target_loss_pct = (u['CURRENT_W'] - u['TARGET_W']) / u['CURRENT_W'] * 100
        
        c1, c2 = st.columns(2)
        with c1: kcal = st.number_input("일일 칼로리(kcal):", 500, 20000, 2200, step=50)
        with c2: sugar = st.number_input("일일 당분(g):", 0, 500, 60, step=1)
        
        c3, c4 = st.columns(2)
        with c3: v_s = st.slider("고강도 운동(주):", 0, 7, int(u['PAQ610']))
        with c4: m_s = st.slider("중강도 운동(주):", 0, 7, int(u['PAQ655']))

        if st.button("🔄 AI 확률 계산", type="primary"):
            with st.spinner("분석 중..."):
                rf, lr, sc, fn = load_assets()
                input_d = u.copy()
                input_d.update({'DR1TKCAL': kcal, 'DR1TSUGR': sugar, 'PAQ610': v_s, 'PAQ655': m_s})
                in_df = pd.DataFrame([{f: input_d.get(f, 0) for f in fn}])[fn]
                p_rf, p_lr = rf.predict_proba(sc.transform(in_df))[0][1]*100, lr.predict_proba(sc.transform(in_df))[0][1]*100
                base_prob = (p_rf * 0.3) + (p_lr * 0.7)
                difficulty_factor = 10 / target_loss_pct 
                sugar_p = (sugar - 60) * 0.2
                v_curve = -1.5 * (v_s - 3.5)**2 + 18
                m_curve = -0.5 * (m_s - 5)**2 + 12
                v_orig_curve = -1.5 * (u['PAQ610'] - 3.5)**2 + 18
                m_orig_curve = -0.5 * (u['PAQ655'] - 5)**2 + 12
                ex_bonus = (v_curve - v_orig_curve) + (m_curve - m_orig_curve)
                st.session_state.current_prob = np.clip((base_prob * difficulty_factor) - sugar_p + ex_bonus, 0, 99.9)

        if 'current_prob' in st.session_state:
            prob = st.session_state.current_prob
            if prob < 20: color, msg_type = "#e74c3c", "error"
            elif prob < 50: color, msg_type = "#f1c40f", "warning"
            elif prob < 80: color, msg_type = "#2ecc71", "info"
            else: color, msg_type = "#27ae60", "success"
            
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: white; border-radius: 15px; border: 3px solid {color}; margin: 15px 0;'>
                    <p style='color: #7f8c8d; font-weight: bold; margin: 0; font-size: 1rem;'>AI 예측 성공 확률</p>
                    <h2 style='color: {color}; margin: 0; font-size: 4rem;'>{prob:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if msg_type == "error": 
                if v_s >= 5 or m_s >= 6:
                    st.error("📉 목표가 매우 높습니다! 운동은 충분하니, 칼로리와 당분 섭취를 조금 더 과감하게 줄여보세요.")
                else:
                    st.error("📉 도전적인 목표! 식단 조절과 운동 빈도를 더 철저히 계획해 보세요.")
            elif msg_type == "warning": 
                if v_s >= 5 or m_s >= 6:
                    st.warning("⚠️ 운동은 이미 완벽한 수준입니다! 이제 식단 일기를 쓰며 식사량을 세밀하게 조절해 볼까요?")
                else:
                    st.warning("⚠️ 조금 더 관리가 필요합니다. 주당 운동 횟수를 1~2회만 더 늘려볼까요?")
            elif msg_type == "info": st.info(f"👍 실현 가능한 목표입니다! 현재 계획을 유지해 주세요.")
            else: st.success(f"🎉 성공이 눈앞에 보입니다! 지금처럼만 계속해 주세요!")
            
            with st.expander("📝 시뮬레이션 알고리즘 가이드"):
                st.write(f"*   **목표 난이도**: 현재 {target_loss_pct:.1f}% 감량 목표 반영")
                st.write(f"*   **운동 효율**: 고강도 주 3.5회, 중강도 주 5회 최적화")
