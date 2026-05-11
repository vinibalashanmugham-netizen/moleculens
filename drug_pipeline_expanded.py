# ==========================================
# INSTALL DEPENDENCIES
# ==========================================
# ============================================================
# ADVANCED QSPR PIPELINE FOR COLAB + GITHUB
# ============================================================
# This code creates separate output pages/files for:
# 1. Drug data extraction or CSV input
# 2. Physicochemical properties
# 3. Degree, degree-sum, and reverse-degree topological indices
# 4. MACCS fingerprints
# 5. Correlation: indices vs properties
# 6. Linear regression models on separate page
# 7. Non-linear regression models on separate page
# 8. Combined model comparison
# 9. VIF analysis
# 10. SHAP analysis
# 11. Y-randomization
# 12. Best model selection
# 13. Full Excel report with separate sheets
# ============================================================

# ============================================================
# COLAB INSTALLATION CELL
# Run this first in Google Colab
# ============================================================
# !pip install rdkit-pypi chembl_webresource_client xgboost lightgbm catboost shap statsmodels openpyxl requests pandas numpy scikit-learn matplotlib seaborn


# ============================================================
# REQUIRED IMPORTS
# ============================================================
import os
import math
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from chembl_webresource_client.new_client import new_client

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, MACCSkeys

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Linear models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    SGDRegressor,
    TweedieRegressor,
    PoissonRegressor,
    GammaRegressor
)

# Non-linear models
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# USER SETTINGS
# ============================================================
OUTPUT_DIR = "QSPR_RESULTS_ADVANCED"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If you want to use your own file, keep USE_CHEMBL = False
# Your CSV must contain a column named SMILES.
USE_CHEMBL = False
INPUT_CSV = "drug_smiles.csv"
DISEASE_NAME = "diabetes"
MAX_DRUGS = 100

RANDOM_STATE = 42
N_SPLITS = 5
N_Y_RANDOMIZATION = 50

PROPERTIES = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "MolMR"
]

INDEX_PREFIXES = ("D_", "DS_", "RD1_", "RD2_", "RD3_", "RD4_")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": math.sqrt(mse),
        "MSE": mse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": safe_mape(y_true, y_pred)
    }


def get_index_columns(df):
    return [col for col in df.columns if col.startswith(INDEX_PREFIXES)]


def get_maccs_columns(df):
    return [col for col in df.columns if col.startswith("MACCS_")]


def clean_numeric_table(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)


# ============================================================
# PAGE 1: DATA INPUT OR CHEMBL EXTRACTION
# ============================================================
def extract_disease_data_from_chembl(disease_name, max_drugs=100):
    print(f"[*] Page 1: Extracting ChEMBL data for disease: {disease_name}")

    indication_api = new_client.drug_indication
    indications = indication_api.filter(mesh_heading__icontains=disease_name).only(["molecule_chembl_id"])

    chembl_ids = list(set([ind["molecule_chembl_id"] for ind in indications]))
    if not chembl_ids:
        print(f"[!] No drugs found for {disease_name}")
        return pd.DataFrame()

    molecule_api = new_client.molecule
    molecules = molecule_api.filter(
        molecule_chembl_id__in=chembl_ids[:max_drugs]
    ).only(["molecule_chembl_id", "pref_name", "molecule_structures"])

    data = []
    for mol in molecules:
        try:
            smiles = mol["molecule_structures"]["canonical_smiles"]
            rd_mol = Chem.MolFromSmiles(smiles)
            if rd_mol is None:
                continue

            data.append({
                "ChEMBL_ID": mol.get("molecule_chembl_id"),
                "Drug_Name": mol.get("pref_name"),
                "SMILES": smiles
            })
        except Exception:
            continue

    df = pd.DataFrame(data)
    df.to_csv(f"{OUTPUT_DIR}/page_1_input_drug_data.csv", index=False)
    return df


def load_input_data():
    if USE_CHEMBL:
        df = extract_disease_data_from_chembl(DISEASE_NAME, MAX_DRUGS)
    else:
        print(f"[*] Page 1: Loading input CSV: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)

    if df.empty:
        raise ValueError("No input data found.")

    if "SMILES" not in df.columns:
        raise ValueError("Input data must contain a column named SMILES")

    df.to_csv(f"{OUTPUT_DIR}/page_1_input_drug_data.csv", index=False)
    return df


# ============================================================
# PAGE 2: PHYSICOCHEMICAL PROPERTIES FROM RDKit
# ============================================================
def compute_rdkit_properties(df):
    print("[*] Page 2: Computing RDKit physicochemical properties...")

    rows = []
    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            rows.append({prop: np.nan for prop in PROPERTIES})
            continue

        rows.append({
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "MolMR": Crippen.MolMR(mol)
        })

    prop_df = pd.DataFrame(rows)

    # Remove old duplicate property columns if they already exist
    df_base = df.drop(columns=[c for c in PROPERTIES if c in df.columns], errors="ignore")
    final_df = pd.concat([df_base.reset_index(drop=True), prop_df.reset_index(drop=True)], axis=1)

    prop_df.to_csv(f"{OUTPUT_DIR}/page_2_properties_only.csv", index=False)
    final_df.to_csv(f"{OUTPUT_DIR}/page_2_dataset_with_properties.csv", index=False)
    return final_df, prop_df


# ============================================================
# OPTIONAL: PUBCHEM PROPERTIES
# ============================================================
PUBCHEM_PROPS = ",".join([
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "ExactMass",
    "MonoisotopicMass",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "IsotopeAtomCount",
    "AtomStereoCount",
    "DefinedAtomStereoCount",
    "UndefinedAtomStereoCount",
    "BondStereoCount",
    "DefinedBondStereoCount",
    "UndefinedBondStereoCount",
    "CovalentUnitCount"
])


def fetch_pubchem_properties(smiles):
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"property/{PUBCHEM_PROPS}/JSON"
        )
        resp = requests.post(url, data={"smiles": smiles}, timeout=20)
        if resp.status_code != 200:
            return {}

        props = resp.json()["PropertyTable"]["Properties"][0]
        return {f"PC_{key}": value for key, value in props.items()}
    except Exception:
        return {}


def enrich_with_pubchem(df):
    print("[*] Optional: Fetching PubChem properties...")

    all_props = []
    for i, smiles in enumerate(df["SMILES"]):
        all_props.append(fetch_pubchem_properties(str(smiles)))
        time.sleep(0.2)
        if (i + 1) % 10 == 0:
            print(f"    PubChem completed: {i + 1}/{len(df)}")

    pubchem_df = pd.DataFrame(all_props)
    final_df = pd.concat([df.reset_index(drop=True), pubchem_df.reset_index(drop=True)], axis=1)
    pubchem_df.to_csv(f"{OUTPUT_DIR}/page_2b_pubchem_properties.csv", index=False)
    return final_df, pubchem_df


# ============================================================
# PAGE 3: TOPOLOGICAL INDICES
# Degree, degree-sum, and reverse-degree indices
# ============================================================
def compute_degree_sums(mol):
    degree_sums = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        degree_sums[idx] = sum(nbr.GetDegree() for nbr in atom.GetNeighbors())
    return degree_sums


def reverse_degree(d, k, delta):
    if delta == 0:
        return 0
    if k <= d:
        return delta - d + k
    value = (delta - d + k) % delta
    return value if value != 0 else delta


def calculate_indices_from_edge_values(edge_values, prefix=""):
    results = {
        f"{prefix}M1": 0,
        f"{prefix}M2": 0,
        f"{prefix}ABC": 0,
        f"{prefix}R": 0,
        f"{prefix}H": 0,
        f"{prefix}F": 0,
        f"{prefix}SDD": 0,
        f"{prefix}SO": 0,
        f"{prefix}GA": 0,
        f"{prefix}GH": 0,
        f"{prefix}HG": 0,
        f"{prefix}ISI": 0,
        f"{prefix}BM": 0,
        f"{prefix}TM": 0,
        f"{prefix}HM": 0,
        f"{prefix}sigma": 0
    }

    for x, y in edge_values:
        if x <= 0 or y <= 0:
            continue

        results[f"{prefix}M1"] += x + y
        results[f"{prefix}M2"] += x * y
        results[f"{prefix}R"] += 1 / math.sqrt(x * y)
        results[f"{prefix}H"] += 2 / (x + y)
        results[f"{prefix}F"] += x**2 + y**2
        results[f"{prefix}SDD"] += (x**2 + y**2) / (x * y)
        results[f"{prefix}SO"] += math.sqrt(x**2 + y**2)
        results[f"{prefix}GA"] += (2 * math.sqrt(x * y)) / (x + y)
        results[f"{prefix}GH"] += ((x + y) * math.sqrt(x * y)) / 2
        results[f"{prefix}HG"] += 2 / ((x + y) * math.sqrt(x * y))
        results[f"{prefix}ISI"] += (x * y) / (x + y)
        results[f"{prefix}BM"] += x + y + x * y
        results[f"{prefix}TM"] += x**2 + y**2 + x * y
        results[f"{prefix}HM"] += (x + y)**2
        results[f"{prefix}sigma"] += (x - y)**2

        if (x + y - 2) > 0:
            results[f"{prefix}ABC"] += math.sqrt((x + y - 2) / (x * y))

    return results


def compute_topological_indices(df):
    print("[*] Page 3: Computing topological indices...")

    all_results = []

    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            all_results.append({})
            continue

        mol = Chem.RemoveHs(mol)
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        delta = max(degrees) if degrees else 0
        degree_sums = compute_degree_sums(mol)

        degree_edges = []
        degree_sum_edges = []

        for bond in mol.GetBonds():
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()

            d1 = a.GetDegree()
            d2 = b.GetDegree()
            degree_edges.append((d1, d2))

            s1 = degree_sums[a.GetIdx()]
            s2 = degree_sums[b.GetIdx()]
            degree_sum_edges.append((s1, s2))

        row_indices = {}
        row_indices.update(calculate_indices_from_edge_values(degree_edges, prefix="D_"))
        row_indices.update(calculate_indices_from_edge_values(degree_sum_edges, prefix="DS_"))

        for k in range(1, 5):
            reverse_edges = []
            for x, y in degree_edges:
                rx = reverse_degree(x, k, delta)
                ry = reverse_degree(y, k, delta)
                reverse_edges.append((rx, ry))
            row_indices.update(calculate_indices_from_edge_values(reverse_edges, prefix=f"RD{k}_"))

        all_results.append(row_indices)

    index_df = pd.DataFrame(all_results)
    final_df = pd.concat([df.reset_index(drop=True), index_df.reset_index(drop=True)], axis=1)

    index_df.to_csv(f"{OUTPUT_DIR}/page_3_topological_indices_only.csv", index=False)
    final_df.to_csv(f"{OUTPUT_DIR}/page_3_dataset_with_topological_indices.csv", index=False)
    return final_df, index_df


# ============================================================
# PAGE 4: MACCS FINGERPRINTS
# ============================================================
def compute_maccs_fingerprints(df):
    print("[*] Page 4: Computing MACCS fingerprints...")

    rows = []
    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            rows.append([np.nan] * 167)
            continue

        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr.tolist())

    maccs_cols = [f"MACCS_{i}" for i in range(167)]
    maccs_df = pd.DataFrame(rows, columns=maccs_cols)
    final_df = pd.concat([df.reset_index(drop=True), maccs_df.reset_index(drop=True)], axis=1)

    maccs_df.to_csv(f"{OUTPUT_DIR}/page_4_maccs_fingerprints_only.csv", index=False)
    final_df.to_csv(f"{OUTPUT_DIR}/page_4_dataset_with_maccs.csv", index=False)
    return final_df, maccs_df


# ============================================================
# PAGE 5: CORRELATION ANALYSIS
# ============================================================
def run_correlation(df):
    print("[*] Page 5: Computing correlation between indices and properties...")

    indices = get_index_columns(df)
    corr = df[indices + PROPERTIES].corr().loc[indices, PROPERTIES]
    corr.to_csv(f"{OUTPUT_DIR}/page_5_indices_properties_correlation.csv")

    plt.figure(figsize=(10, max(6, len(indices) * 0.15)))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(PROPERTIES)), PROPERTIES, rotation=45, ha="right")
    plt.yticks(range(len(indices)), indices, fontsize=6)
    plt.title("Correlation between topological indices and physicochemical properties")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/page_5_correlation_heatmap.png", dpi=300)
    plt.close()

    return corr


# ============================================================
# PAGE 6: LINEAR REGRESSION MODELS
# ============================================================
def get_linear_models():
    return {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression Degree 2": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=20000))
        ]),
        "ElasticNet Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000))
        ]),
        "Bayesian Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge())
        ]),
        "Huber Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(max_iter=2000))
        ]),
        "Theil-Sen Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", TheilSenRegressor(random_state=RANDOM_STATE, max_subpopulation=10000))
        ]),
        "RANSAC Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RANSACRegressor(random_state=RANDOM_STATE))
        ]),
        "Orthogonal Matching Pursuit": Pipeline([
            ("scaler", StandardScaler()),
            ("model", OrthogonalMatchingPursuit())
        ]),
        "Passive Aggressive Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", PassiveAggressiveRegressor(random_state=RANDOM_STATE, max_iter=2000))
        ]),
        "SGD Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SGDRegressor(random_state=RANDOM_STATE, max_iter=3000))
        ]),
        "Tweedie Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", TweedieRegressor(power=0, alpha=0.5, max_iter=2000))
        ])
    }


# ============================================================
# PAGE 7: NON-LINEAR REGRESSION MODELS
# ============================================================
def get_nonlinear_models():
    return {
        "SVR RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, gamma="scale"))
        ]),
        "SVR Polynomial": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="poly", degree=2, C=10, gamma="scale"))
        ]),
        "KNN Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
        "Bagging Regressor": BaggingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective="reg:squarederror"
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            verbose=0,
            random_state=RANDOM_STATE
        ),
        "Neural Network MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),
        "Gaussian Process RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=C(1.0) * RBF(length_scale=1.0),
                random_state=RANDOM_STATE,
                normalize_y=True
            ))
        ])
    }


# ============================================================
# GENERAL MODEL RUNNER
# ============================================================
def run_model_group(df, models, feature_cols, page_name, model_group):
    print(f"[*] {page_name}: Running {model_group} models...")

    X = clean_numeric_table(df[feature_cols])
    rows = []
    prediction_rows = []
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for prop in PROPERTIES:
        y = df[prop].values

        for model_name, model in models.items():
            try:
                y_pred = cross_val_predict(model, X, y, cv=cv)
                metrics = regression_metrics(y, y_pred)

                rows.append({
                    "Model_Group": model_group,
                    "Property": prop,
                    "Model": model_name,
                    "Feature_Count": len(feature_cols),
                    **metrics
                })

                for i, (actual, pred) in enumerate(zip(y, y_pred)):
                    prediction_rows.append({
                        "Model_Group": model_group,
                        "Property": prop,
                        "Model": model_name,
                        "Sample": i + 1,
                        "Actual": actual,
                        "Predicted": pred
                    })
            except Exception as e:
                rows.append({
                    "Model_Group": model_group,
                    "Property": prop,
                    "Model": model_name,
                    "Feature_Count": len(feature_cols),
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "Error": str(e)
                })

    results_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(prediction_rows)

    safe_page = page_name.lower().replace(" ", "_").replace(":", "")
    results_df.to_csv(f"{OUTPUT_DIR}/{safe_page}_metrics.csv", index=False)
    pred_df.to_csv(f"{OUTPUT_DIR}/{safe_page}_actual_predicted.csv", index=False)

    return results_df, pred_df


# ============================================================
# PAGE 6A: LINEAR MODELS USING TOPOLOGICAL INDICES
# PAGE 6B: LINEAR MODELS USING MACCS
# PAGE 6C: LINEAR MODELS USING INDICES + MACCS
# ============================================================
def run_all_linear_pages(df):
    linear_models = get_linear_models()
    index_cols = get_index_columns(df)
    maccs_cols = get_maccs_columns(df)
    combined_cols = index_cols + maccs_cols

    lin_indices, pred_lin_indices = run_model_group(
        df, linear_models, index_cols,
        "page_6a_linear_models_indices", "Linear_Indices"
    )
    lin_maccs, pred_lin_maccs = run_model_group(
        df, linear_models, maccs_cols,
        "page_6b_linear_models_maccs", "Linear_MACCS"
    )
    lin_combined, pred_lin_combined = run_model_group(
        df, linear_models, combined_cols,
        "page_6c_linear_models_indices_plus_maccs", "Linear_Indices_MACCS"
    )

    return {
        "linear_indices": lin_indices,
        "linear_maccs": lin_maccs,
        "linear_combined": lin_combined,
        "pred_linear_indices": pred_lin_indices,
        "pred_linear_maccs": pred_lin_maccs,
        "pred_linear_combined": pred_lin_combined
    }


# ============================================================
# PAGE 7A: NON-LINEAR MODELS USING TOPOLOGICAL INDICES
# PAGE 7B: NON-LINEAR MODELS USING MACCS
# PAGE 7C: NON-LINEAR MODELS USING INDICES + MACCS
# ============================================================
def run_all_nonlinear_pages(df):
    nonlinear_models = get_nonlinear_models()
    index_cols = get_index_columns(df)
    maccs_cols = get_maccs_columns(df)
    combined_cols = index_cols + maccs_cols

    nonlin_indices, pred_nonlin_indices = run_model_group(
        df, nonlinear_models, index_cols,
        "page_7a_nonlinear_models_indices", "Nonlinear_Indices"
    )
    nonlin_maccs, pred_nonlin_maccs = run_model_group(
        df, nonlinear_models, maccs_cols,
        "page_7b_nonlinear_models_maccs", "Nonlinear_MACCS"
    )
    nonlin_combined, pred_nonlin_combined = run_model_group(
        df, nonlinear_models, combined_cols,
        "page_7c_nonlinear_models_indices_plus_maccs", "Nonlinear_Indices_MACCS"
    )

    return {
        "nonlinear_indices": nonlin_indices,
        "nonlinear_maccs": nonlin_maccs,
        "nonlinear_combined": nonlin_combined,
        "pred_nonlinear_indices": pred_nonlin_indices,
        "pred_nonlinear_maccs": pred_nonlin_maccs,
        "pred_nonlinear_combined": pred_nonlin_combined
    }


# ============================================================
# PAGE 8: COMBINED LINEAR + NON-LINEAR COMPARISON
# ============================================================
def make_combined_model_comparison(linear_results, nonlinear_results):
    print("[*] Page 8: Creating combined model comparison...")

    metric_tables = [
        linear_results["linear_indices"],
        linear_results["linear_maccs"],
        linear_results["linear_combined"],
        nonlinear_results["nonlinear_indices"],
        nonlinear_results["nonlinear_maccs"],
        nonlinear_results["nonlinear_combined"]
    ]

    combined = pd.concat(metric_tables, ignore_index=True)
    combined = combined.sort_values(["Property", "R2", "RMSE"], ascending=[True, False, True])
    combined.to_csv(f"{OUTPUT_DIR}/page_8_all_linear_and_nonlinear_model_comparison.csv", index=False)
    return combined


# ============================================================
# PAGE 9: VIF ANALYSIS FOR TOPOLOGICAL INDICES
# ============================================================
def run_vif_analysis(df):
    print("[*] Page 9: Running VIF analysis...")

    feature_cols = get_index_columns(df)
    X = clean_numeric_table(df[feature_cols])
    X = X.loc[:, X.nunique() > 1]

    rows = []
    for i, col in enumerate(X.columns):
        try:
            vif_value = variance_inflation_factor(X.values, i)
        except Exception:
            vif_value = np.nan
        rows.append({"Feature": col, "VIF": vif_value})

    vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(f"{OUTPUT_DIR}/page_9_vif_analysis.csv", index=False)
    return vif_df


# ============================================================
# PAGE 10: SHAP ANALYSIS
# Uses Random Forest because it is stable for small QSPR datasets
# ============================================================
def run_shap_analysis(df):
    print("[*] Page 10: Running SHAP analysis...")

    feature_cols = get_index_columns(df)
    X = clean_numeric_table(df[feature_cols])
    shap_tables = []

    for prop in PROPERTIES:
        y = df[prop].values
        model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        prop_shap_df = pd.DataFrame({
            "Property": prop,
            "Feature": feature_cols,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values("Mean_Abs_SHAP", ascending=False)

        prop_shap_df.to_csv(f"{OUTPUT_DIR}/page_10_shap_importance_{prop}.csv", index=False)
        shap_tables.append(prop_shap_df)

        try:
            shap.summary_plot(shap_values, X, show=False, max_display=20)
            plt.title(f"SHAP summary for {prop}")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/page_10_shap_summary_{prop}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    shap_all = pd.concat(shap_tables, ignore_index=True)
    shap_all.to_csv(f"{OUTPUT_DIR}/page_10_shap_all_properties.csv", index=False)
    return shap_all


# ============================================================
# PAGE 11: Y-RANDOMIZATION TEST
# ============================================================
def run_y_randomization(df, n_random=N_Y_RANDOMIZATION):
    print("[*] Page 11: Running Y-randomization...")

    feature_cols = get_index_columns(df)
    X = clean_numeric_table(df[feature_cols])
    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    for prop in PROPERTIES:
        y = df[prop].values

        original_pred = cross_val_predict(model, X, y, cv=cv)
        original_metrics = regression_metrics(y, original_pred)

        rows.append({
            "Property": prop,
            "Run_Type": "Original",
            "Run": 0,
            **original_metrics
        })

        for r in range(1, n_random + 1):
            rng = np.random.RandomState(1000 + r)
            y_random = rng.permutation(y)
            y_random_pred = cross_val_predict(model, X, y_random, cv=cv)
            random_metrics = regression_metrics(y_random, y_random_pred)

            rows.append({
                "Property": prop,
                "Run_Type": "Randomized",
                "Run": r,
                **random_metrics
            })

    yrand_df = pd.DataFrame(rows)
    yrand_df.to_csv(f"{OUTPUT_DIR}/page_11_y_randomization.csv", index=False)

    for prop in PROPERTIES:
        temp = yrand_df[yrand_df["Property"] == prop]
        original = temp[temp["Run_Type"] == "Original"]["R2"].values
        randomized = temp[temp["Run_Type"] == "Randomized"]["R2"].values

        plt.figure(figsize=(6, 4))
        plt.boxplot([randomized], labels=["Randomized R2"])
        plt.scatter([1], original, marker="o", label="Original R2")
        plt.ylabel("R2")
        plt.title(f"Y-randomization for {prop}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/page_11_y_randomization_{prop}.png", dpi=300)
        plt.close()

    return yrand_df


# ============================================================
# PAGE 12: BEST MODEL SELECTION
# ============================================================
def select_best_models(combined_metrics):
    print("[*] Page 12: Selecting best models...")

    valid = combined_metrics.dropna(subset=["R2", "RMSE", "MAE"]).copy()

    best_rows = []
    for prop in PROPERTIES:
        temp = valid[valid["Property"] == prop].copy()
        temp = temp.sort_values(["R2", "RMSE", "MAE"], ascending=[False, True, True])
        if len(temp) > 0:
            best_rows.append(temp.iloc[0])

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(f"{OUTPUT_DIR}/page_12_best_models_per_property.csv", index=False)
    return best_df


# ============================================================
# PAGE 13: ACTUAL VS PREDICTED PLOTS FOR BEST MODELS
# ============================================================
def make_actual_predicted_plots(df, best_df):
    print("[*] Page 13: Creating actual vs predicted plots for best models...")

    all_models = {}
    all_models.update(get_linear_models())
    all_models.update(get_nonlinear_models())

    index_cols = get_index_columns(df)
    maccs_cols = get_maccs_columns(df)

    for _, row in best_df.iterrows():
        prop = row["Property"]
        model_name = row["Model"]
        group = row["Model_Group"]

        if "MACCS" in group and "Indices_MACCS" not in group:
            feature_cols = maccs_cols
        elif "Indices_MACCS" in group:
            feature_cols = index_cols + maccs_cols
        else:
            feature_cols = index_cols

        X = clean_numeric_table(df[feature_cols])
        y = df[prop].values
        model = all_models[model_name]
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        try:
            pred = cross_val_predict(model, X, y, cv=cv)

            plt.figure(figsize=(5, 5))
            plt.scatter(y, pred)
            min_val = min(np.min(y), np.min(pred))
            max_val = max(np.max(y), np.max(pred))
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{prop}: {model_name}")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/page_13_actual_vs_predicted_{prop}.png", dpi=300)
            plt.close()
        except Exception:
            pass


# ============================================================
# PAGE 14: SAVE COMPLETE EXCEL REPORT
# ============================================================
def save_excel_report(results_dict):
    print("[*] Page 14: Saving Excel report with separate sheets...")

    excel_path = f"{OUTPUT_DIR}/QSPR_advanced_full_report.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, table in results_dict.items():
            if isinstance(table, pd.DataFrame):
                safe_sheet = sheet_name[:31]
                table.to_excel(writer, sheet_name=safe_sheet, index=False)

    print(f"✅ Excel report saved: {excel_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    df = load_input_data()

    # Page 2: Properties
    df, properties_only = compute_rdkit_properties(df)

    # Optional PubChem page
    # Uncomment the next line if you want PubChem properties also.
    # df, pubchem_df = enrich_with_pubchem(df)

    # Page 3: Topological indices
    df, indices_only = compute_topological_indices(df)

    # Page 4: MACCS fingerprints
    df, maccs_only = compute_maccs_fingerprints(df)

    # Page 5: Correlation
    correlation_df = run_correlation(df)

    # Pages 6 and 7: Linear and non-linear models separately
    linear_results = run_all_linear_pages(df)
    nonlinear_results = run_all_nonlinear_pages(df)

    # Page 8: Combined comparison
    combined_metrics = make_combined_model_comparison(linear_results, nonlinear_results)

    # Page 9: VIF
    vif_df = run_vif_analysis(df)

    # Page 10: SHAP
    shap_df = run_shap_analysis(df)

    # Page 11: Y-randomization
    yrand_df = run_y_randomization(df)

    # Page 12: Best model selection
    best_df = select_best_models(combined_metrics)

    # Page 13: Plots
    make_actual_predicted_plots(df, best_df)

    # Final complete dataset
    df.to_csv(f"{OUTPUT_DIR}/complete_dataset_properties_indices_maccs.csv", index=False)

    # Page 14: Full Excel report
    save_excel_report({
        "Input_Data": df[[c for c in df.columns if c not in get_index_columns(df) + get_maccs_columns(df)]],
        "Properties": properties_only,
        "Topological_Indices": indices_only,
        "MACCS_Fingerprints": maccs_only,
        "Correlation": correlation_df.reset_index().rename(columns={"index": "Topological_Index"}),
        "Linear_Indices": linear_results["linear_indices"],
        "Linear_MACCS": linear_results["linear_maccs"],
        "Linear_Combined": linear_results["linear_combined"],
        "Nonlinear_Indices": nonlinear_results["nonlinear_indices"],
        "Nonlinear_MACCS": nonlinear_results["nonlinear_maccs"],
        "Nonlinear_Combined": nonlinear_results["nonlinear_combined"],
        "All_Model_Comparison": combined_metrics,
        "VIF": vif_df,
        "SHAP": shap_df,
        "Y_Randomization": yrand_df,
        "Best_Models": best_df
    })

    print("\n✅ ADVANCED QSPR PIPELINE COMPLETED")
    print(f"✅ All separate pages/files are saved inside: {OUTPUT_DIR}")
    print("✅ Main Excel file: QSPR_advanced_full_report.xlsx")

    return df, combined_metrics, best_df


# ============================================================
# RUN CODE
# ============================================================
# For GitHub / VS Code:
# 1. Keep drug_smiles.csv in the same folder.
# 2. Run: python qspr_advanced_pipeline.py

# For Google Colab:
# 1. Run the pip install cell.
# 2. Upload drug_smiles.csv.
# 3. Run all cells.

if __name__ == "__main__":
    df_final, all_results, best_models = main()
