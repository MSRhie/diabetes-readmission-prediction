"""
당뇨 환자 30일 내 재입원 예측 모델
=====================================
Strack et al. (2014) 논문 재현 + XGBoost + SHAP 해석

Dataset  : UCI Diabetes 130-US Hospitals (brandao/diabetes on Kaggle)
Reference: Strack et al. (2014), Impact of HbA1c Measurement on
           Hospital Readmission Rates: Analysis of 70,000 Clinical
           Database Patient Records

Author   : Minshik Rhie
GitHub   : https://github.com/MSRhie
"""

# ── 0. 라이브러리 ─────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import kagglehub
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')


# ── 1. 데이터 로드 ────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Kaggle에서 당뇨 데이터셋을 다운로드하고 로드한다."""
    path = kagglehub.dataset_download("brandao/diabetes")
    df = pd.read_csv(os.path.join(path, 'diabetic_data.csv'))
    print(f"데이터 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


# ── 2. 상수 정의 ──────────────────────────────────────────────────
CATEGORICAL_COLS = [
    'race', 'gender', 'age',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'diag_1', 'diag_2', 'diag_3',
    'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed', 'payer_code', 'medical_specialty'
]

DISCRETE_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

ID_COLS   = ['encounter_id', 'patient_nbr']
TARGET    = 'readmitted'

DROP_COLS         = ['weight', 'payer_code', 'encounter_id', 'patient_nbr']
BINARY_MISSING_COLS = ['medical_specialty', 'max_glu_serum', 'A1Cresult']
MISSING_VALUES    = ['?', 'Unknown/Invalid']

# 재입원 불가 퇴원 형태 (사망, 호스피스)
EXCLUDE_DISCHARGE = [11, 13, 19, 20, 21]

# 로지스틱 회귀용 로그 변환 대상
LOG_COLS = [
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'num_medications', 'time_in_hospital'
]

# XGBoost 사용 시 제거할 컬럼 (원본 문자열 + 타겟 원본)
DROP_FOR_XGB = ['readmitted', 'medical_specialty', 'max_glu_serum', 'A1Cresult']


# ── 3. 전처리 함수 ────────────────────────────────────────────────
def drop_duplicates_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    환자당 첫 번째 입원 기록만 유지한다.

    동일 환자의 반복 입원 기록은 로지스틱 회귀의 독립성 가정을 위반하므로
    논문(Strack et al., 2014)과 동일하게 첫 번째 기록만 사용한다.
    """
    result = (
        df
        .sort_values(['encounter_id'])
        .drop_duplicates(['patient_nbr'], keep='first')
        .reset_index(drop=True)
    )
    print(f"중복 제거 후: {result.shape[0]:,}행")
    return result


def transform_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """'?', 'Unknown/Invalid'를 NaN으로 변환한다."""
    return df.replace(MISSING_VALUES, np.nan)


def exclude_no_readmission_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    재입원이 불가능한 상태의 환자를 제거한다.
    - 11: 사망, 13/19/20/21: 호스피스
    """
    result = df[~df['discharge_disposition_id'].isin(EXCLUDE_DISCHARGE)].copy()
    print(f"재입원 불가 케이스 제거 후: {result.shape[0]:,}행")
    return result


def create_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측 여부를 binary 변수로 생성하고 불필요한 변수를 제거한다.

    medical_specialty, max_glu_serum, A1Cresult는 결측 여부 자체가
    케어 적극성을 반영(MNAR 가능성)하므로 binary 변수로 변환한다.
    결측값 대체(imputation)는 수행하지 않는다.
    """
    result = df.copy()
    for col in BINARY_MISSING_COLS:
        result[f'{col}_missing'] = np.where(result[col].isnull(), 0, 1)
    result['target'] = np.where(result[TARGET].eq('<30'), 1, 0)
    result.drop(columns=DROP_COLS, inplace=True)
    print(f"전처리 후: {result.shape[1]}개 컬럼")
    return result


def classify_icd9(col_var: str, df: pd.DataFrame) -> pd.Series:
    """
    ICD-9 코드를 논문(Strack et al., 2014) Table 2 기준 9개 카테고리로 분류한다.

    Categories: Circulatory, Respiratory, Digestive, Diabetes,
                Injury, Musculoskeletal, Genitourinary, Neoplasms, Other
    """
    code = df[col_var].astype(str)
    numeric_code = pd.to_numeric(
        code.str.replace('E', '350', regex=False)
            .str.replace('V', '999', regex=False),
        errors='coerce'
    )

    conditions = [
        numeric_code.between(390, 459) | (numeric_code == 785),
        numeric_code.between(460, 519) | (numeric_code == 786),
        numeric_code.between(520, 579) | (numeric_code == 787),
        code.str.startswith('250'),
        numeric_code.between(800, 999),
        numeric_code.between(710, 739),
        numeric_code.between(580, 629) | (numeric_code == 788),
        numeric_code.between(140, 239),
    ]
    choices = [
        'Circulatory', 'Respiratory', 'Digestive', 'Diabetes',
        'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms'
    ]
    return np.select(conditions, choices, default='Other')


def encode_drug_cols(df: pd.DataFrame, drug_cols: list) -> pd.DataFrame:
    """
    약물 변수를 처방 여부 binary 변수(_YN)로 변환한다.
    Steady/Down/Up → 1 (처방 있음), No → 0 (처방 없음)
    """
    result = df.copy()
    for col in drug_cols:
        if col in result.columns:
            result[f'{col}_YN'] = (result[col] != 'No').astype(int)
    return result


def preprocess_common(df: pd.DataFrame) -> pd.DataFrame:
    """공통 전처리 파이프라인을 실행한다."""
    df = drop_duplicates_rows(df)
    df = transform_missing_values(df)
    df = exclude_no_readmission_cases(df)
    df = create_missing_features(df)

    # ICD-9 재범주화
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        df[f'{diag}_cate'] = classify_icd9(diag, df)
        df.drop(columns=[diag], inplace=True)

    # 약물 binary 변환
    drug_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    df = encode_drug_cols(df, drug_cols)

    # 더미변수화
    nominal_cols = [
        'race', 'gender', 'age', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'diag_1_cate', 'diag_2_cate', 'diag_3_cate'
    ]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"공통 전처리 완료: {df.shape[1]}개 컬럼")
    return df


# ── 4. 로지스틱 회귀용 전처리 ─────────────────────────────────────
def preprocess_logistic(df: pd.DataFrame) -> pd.DataFrame:
    """
    로지스틱 회귀용 추가 전처리를 수행한다.

    - 로그 변환 (zero-inflated 변수)
    - 파생변수 생성 (lab_per_day)
    - 다중공선성 처리 (VIF 기반)
    - 희소 변수 제거 (분산 기반)
    - 표준화 (수치형 변수만)
    """
    df_lr = df.copy()

    # 문자열 컬럼 제거 (원본 남아있는 경우)
    obj_cols = df_lr.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        df_lr.drop(columns=obj_cols, inplace=True)

    # 로그 변환 (log1p: 0 포함 변수 대응)
    existing_log_cols = [c for c in LOG_COLS if c in df_lr.columns]
    df_lr[existing_log_cols] = df_lr[existing_log_cols].apply(np.log1p)

    # 파생변수: 하루당 검사 수 (다중공선성 해소)
    df_lr['lab_per_day'] = df_lr['num_lab_procedures'] / df_lr['time_in_hospital']
    df_lr.drop(columns=['num_lab_procedures'], inplace=True)

    # 다중공선성 처리 (VIF 기반)
    # num_medications(VIF 29), number_diagnoses(VIF 14.5): 로지스틱 제거
    # examide_YN, citoglipton_YN, glimepiride-pioglitazone_YN: NaN (완전 다중공선성)
    remove_vif = [
        'num_medications', 'number_diagnoses',
        'examide_YN', 'citoglipton_YN', 'glimepiride-pioglitazone_YN'
    ]
    remove_vif = [c for c in remove_vif if c in df_lr.columns]
    df_lr.drop(columns=remove_vif, inplace=True)

    # 희소 약물 변수 제거 (분산 기반 필터링 결과)
    sparse_drug_cols = [
        'nateglinide_YN', 'chlorpropamide_YN', 'acetohexamide_YN',
        'tolbutamide_YN', 'acarbose_YN', 'miglitol_YN', 'troglitazone_YN',
        'tolazamide_YN', 'glyburide-metformin_YN', 'glipizide-metformin_YN',
        'metformin-rosiglitazone_YN', 'metformin-pioglitazone_YN'
    ]
    sparse_drug_cols = [c for c in sparse_drug_cols if c in df_lr.columns]
    df_lr.drop(columns=sparse_drug_cols, inplace=True)

    # 수치형 변수 표준화 (binary 변수 제외)
    numeric_cols = [
        'time_in_hospital', 'num_procedures', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'lab_per_day'
    ]
    numeric_cols = [c for c in numeric_cols if c in df_lr.columns]
    scaler = StandardScaler()
    df_lr[numeric_cols] = scaler.fit_transform(df_lr[numeric_cols])

    print(f"로지스틱 전처리 완료: {df_lr.shape[1]}개 컬럼")
    return df_lr


# ── 5. XGBoost용 전처리 ───────────────────────────────────────────
def preprocess_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost용 전처리를 수행한다.
    트리 기반 모델은 다중공선성 영향을 받지 않으므로
    로그 변환, 표준화, 다중공선성 처리 없이 사용한다.
    """
    df_xgb = df.copy()

    # 원본 문자열 컬럼 및 타겟 원본 제거
    drop_cols = [c for c in DROP_FOR_XGB if c in df_xgb.columns]
    df_xgb.drop(columns=drop_cols, inplace=True)

    # 컬럼명 특수문자 처리 ([, ] → (, ))
    df_xgb.columns = (
        df_xgb.columns
        .str.replace('[', '(', regex=False)
        .str.replace(']', ')', regex=False)
        .str.replace('<', 'lt_', regex=False)
    )

    print(f"XGBoost 전처리 완료: {df_xgb.shape[1]}개 컬럼")
    return df_xgb


# ── 6. 모델 학습 및 평가 ──────────────────────────────────────────
def train_logistic(X_train, y_train):
    """로지스틱 회귀 모델을 학습한다."""
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_base(X_train, y_train, scale):
    """기본 파라미터 XGBoost 모델을 학습한다."""
    model = XGBClassifier(
        learning_rate=0.3,
        n_estimators=1000,
        early_stopping_rounds=50,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=scale,
        eval_metric='auc',
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def tune_xgboost(X_train, y_train, scale):
    """
    Random Search로 XGBoost 최적 파라미터를 탐색하고 최종 모델을 학습한다.

    최적 파라미터 (탐색 결과):
    - learning_rate: 0.01
    - max_depth: 7
    - subsample: 0.6
    - colsample_bytree: 0.7
    - min_child_weight: 20
    """
    xgb_base = XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale,
        eval_metric='auc',
        random_state=42
    )
    param_dist = {
        'learning_rate'   : [0.01, 0.05, 0.1, 0.3],
        'max_depth'       : [3, 4, 5, 6, 7, 8],
        'subsample'       : [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 5, 10, 20]
    }
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist,
        n_iter=50, cv=5,
        scoring='roc_auc',
        random_state=42, n_jobs=-1, verbose=1
    )
    random_search.fit(X_train, y_train)
    print(f"Best params: {random_search.best_params_}")
    print(f"Best AUC (CV): {random_search.best_score_:.4f}")

    # 최적 파라미터로 최종 모델 학습
    xgb_final = XGBClassifier(
        **random_search.best_params_,
        n_estimators=1000,
        early_stopping_rounds=50,
        scale_pos_weight=scale,
        eval_metric='auc',
        random_state=42
    )
    xgb_final.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return xgb_final


def evaluate_model(model, X_test, y_test, model_name: str):
    """모델 성능을 평가하고 결과를 출력한다."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"[{model_name}] 성능 평가")
    print('='*50)
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {auc:.4f}")
    return auc, y_pred, y_prob


# ── 7. SHAP 해석 ──────────────────────────────────────────────────
def explain_with_shap(model, X_test, patient_idx: int = 0):
    """
    SHAP으로 XGBoost 모델을 해석한다.

    - Bar plot: 전체 변수 중요도
    - Beeswarm plot: 방향성 + 크기
    - Waterfall plot: 개별 환자 해석
    """
    import matplotlib
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("\n[SHAP] 변수 중요도 (Bar plot)")
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=True)

    print("\n[SHAP] Beeswarm plot (방향성 + 크기)")
    shap.summary_plot(shap_values, X_test, show=True)

    print(f"\n[SHAP] 개별 환자 해석 (환자 인덱스: {patient_idx})")
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[patient_idx],
            base_values   = explainer.expected_value,
            data          = X_test.iloc[patient_idx],
            feature_names = X_test.columns.tolist()
        )
    )
    return explainer, shap_values


# ── 8. 메인 실행 ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("당뇨 환자 30일 내 재입원 예측 모델")
    print("Strack et al. (2014) 논문 재현 + XGBoost + SHAP")
    print("=" * 60)

    # 데이터 로드
    df = load_data()

    # 공통 전처리
    df = preprocess_common(df)

    # ── 로지스틱 회귀 ──────────────────────────────────────────
    print("\n" + "─" * 40)
    print("I. 로지스틱 회귀 모델링")
    print("─" * 40)

    df_lr = preprocess_logistic(df.copy())
    X_lr  = df_lr.drop(columns=['target'])
    y_lr  = df_lr['target']

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr
    )
    lr_model = train_logistic(X_train_lr, y_train_lr)
    lr_auc, _, _ = evaluate_model(lr_model, X_test_lr, y_test_lr, "로지스틱 회귀")

    # 회귀계수 및 오즈비
    coef_df = pd.DataFrame({
        '변수': X_train_lr.columns,
        '계수': lr_model.coef_[0]
    }).sort_values('계수', ascending=False)
    coef_df['오즈비'] = np.exp(coef_df['계수'])
    print("\n[로지스틱] 재입원 높이는 상위 10개:")
    print(coef_df.head(10).to_string(index=False))
    print("\n[로지스틱] 재입원 낮추는 하위 10개:")
    print(coef_df.tail(10).to_string(index=False))

    # ── XGBoost ────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("II. XGBoost 모델링")
    print("─" * 40)

    df_xgb = preprocess_xgboost(df.copy())
    X_xgb  = df_xgb.drop(columns=['target'])
    y_xgb  = df_xgb['target']

    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
    )
    scale = y_train_xgb.value_counts()[0] / y_train_xgb.value_counts()[1]

    # 기본 모델
    xgb_base_model = train_xgboost_base(X_train_xgb, y_train_xgb, scale)
    xgb_base_auc, _, _ = evaluate_model(
        xgb_base_model, X_test_xgb, y_test_xgb, "XGBoost (기본)"
    )

    # 튜닝 모델
    print("\n[XGBoost] Random Search 튜닝 중...")
    xgb_final = tune_xgboost(X_train_xgb, y_train_xgb, scale)
    xgb_final_auc, _, _ = evaluate_model(
        xgb_final, X_test_xgb, y_test_xgb, "XGBoost (튜닝)"
    )

    # ── 결과 요약 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 성능 비교")
    print("=" * 60)
    print(f"{'모델':<20} {'AUC-ROC':>10}")
    print("-" * 32)
    print(f"{'로지스틱 회귀':<20} {lr_auc:>10.4f}")
    print(f"{'XGBoost (기본)':<20} {xgb_base_auc:>10.4f}")
    print(f"{'XGBoost (튜닝)':<20} {xgb_final_auc:>10.4f}")
    print(f"{'논문 (Strack 2014)':<20} {'0.6670':>10}")

    # ── SHAP 해석 ──────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("III. SHAP 모델 해석")
    print("─" * 40)
    explain_with_shap(xgb_final, X_test_xgb, patient_idx=0)


if __name__ == "__main__":
    main()
