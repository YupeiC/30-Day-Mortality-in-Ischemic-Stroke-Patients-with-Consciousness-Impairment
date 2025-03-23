import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

# 设置页面标题
st.set_page_config(page_title="30-Day Mortality Risk Prediction Model for ICU Ischemic Stroke Patients with Consciousness Impairment", layout="wide")

# 加载模型
@st.cache_resource
def load_model():
    model = load('CatBoost.pkl')
    return model

model = load_model()

# 创建页面标题
st.title("30-Day Mortality Risk Prediction Model for ICU Ischemic Stroke Patients with Consciousness Impairment")
st.write("Please input patient's clinical indicators:")

# 创建输入字段
col1, col2, col3 = st.columns(3)

with col1:
    ag = st.number_input('Anion Gap', value=15.0, format="%.1f")
    ast = st.number_input('AST', value=27.0, format="%.1f")
    age = st.number_input('Age', value=70, format="%d")

with col2:
    ca = st.number_input('Calcium (Ca)', value=8.7, format="%.1f")
    dbp = st.number_input('Diastolic BP', value=52.0, format="%.1f")
    glucose = st.number_input('Glucose', value=112.0, format="%.1f")

with col3:
    na = st.number_input('Sodium (Na)', value=160.0, format="%.1f")
    rr = st.number_input('Respiratory Rate', value=15.0, format="%.1f")
    sbp = st.number_input('Systolic BP', value=94.0, format="%.1f")

# 创建预测按钮
if st.button('Predict'):
    # 准备输入数据
    input_data = pd.DataFrame({
        'AG': [ag],
        'AST': [ast],
        'Age': [age],
        'Ca': [ca],
        'DBP': [dbp],
        'Glucose': [glucose],
        'Na': [na],
        'RR': [rr],
        'SBP': [sbp]
    })

    # 进行预测
    prediction = model.predict_proba(input_data)[0]
    
    # 显示预测结果
    st.write("---")
    st.subheader("Prediction Results")
    
    # 使用进度条显示死亡风险
    st.metric(
        label="30-Day Mortality Risk",
        value=f"{prediction[1]:.1%}",
        delta=None
    )
    
    # 显示风险等级
    risk_level = "High Risk" if prediction[1] > 0.5 else "Low Risk"
    st.info(f"Risk Level: {risk_level}")

    # SHAP值解释
    st.write("---")
    st.subheader("Model Interpretation")
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # 绘制force plot
    plt.figure(figsize=(15, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # 显示特征重要性说明
    st.write("---")
    st.subheader("Feature Contribution Analysis")
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'SHAP Value': np.abs(shap_values[0])
    }).sort_values('SHAP Value', ascending=False)
    
    # 显示特征重要性表格
    st.table(feature_importance)

# 添加说明信息
st.write("---")
st.markdown("""
### Instructions:
1. Enter the patient's clinical indicators in the input fields above
2. Click the "Predict" button to get results
3. The system will display the 30-day mortality risk and risk level
4. SHAP values show how each feature contributes to the prediction
""")

# 添加模型信息
st.sidebar.title("Model Information")
st.sidebar.info("""
- Model Type: CatBoost Classifier
- Training Data: ICU Patient Clinical Data
- Target Variable: 30-Day Mortality
- Number of Features: 9 Clinical Indicators
""")

# 添加特征说明
st.sidebar.title("Feature Description")
st.sidebar.markdown("""
- AG: Anion Gap
- AST: Aspartate Aminotransferase
- Age: Patient Age
- Ca: Calcium
- DBP: Diastolic Blood Pressure
- Glucose: Blood Glucose
- Na: Sodium
- RR: Respiratory Rate
- SBP: Systolic Blood Pressure
""")