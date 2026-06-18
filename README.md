# Churn prediction MLOps

Projet MLOps de prediction de churn client avec :

- un pipeline d'entrainement scikit-learn ;
- un suivi d'experience MLflow ;
- une API FastAPI protegee par token ;
- des predictions unitaires et batch ;
- des tests pytest.

## Installation

### Windows

```bash
python -m venv env
.\env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Creer un fichier `.env` a la racine du projet a partir de `.env_Exemple` :

```env
API_TOKEN=Votre_clé
```

Le token est obligatoire pour les routes protegees :

- `/predict`
- `/predict_batch`
- `/metrics`

Le fichier `.env` ne doit pas etre versionne.

## Execution du pipeline

```bash
python main.py
```

Le pipeline :

1. lit `data/raw/churn.csv` ;
2. prepare les donnees ;
3. entraine un `RandomForestClassifier` ;
4. evalue le modele ;
5. ecrit les artefacts dans `artifacts/`.

Artefacts attendus :

- `artifacts/model.pkl`
- `artifacts/model.pkl.sha256`
- `artifacts/metrics.json`
- `artifacts/feature_columns.json`

Le fichier `model.pkl.sha256` sert a verifier l'integrite du modele avant son chargement par l'API.

## Lancement de l'API

```bash
uvicorn app:app --reload --port 8001
```

Routes disponibles :

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /predict_batch`

## Healthcheck

```bash
curl http://127.0.0.1:8001/health
```

Si les artefacts sont charges :

```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_columns_loaded": true
}
```

Si le modele ou les colonnes ne sont pas disponibles, l'API retourne HTTP `503`.

## Prediction unitaire

```bash
curl -X POST http://127.0.0.1:8001/predict ^
  -H "Content-Type: application/json" ^
  -H "X-API-Token: Votre_clé" ^
  -d "{\"tenure_months\":12,\"monthly_charges\":75.5,\"total_charges\":906.0,\"contract\":\"Month-to-month\"}"
```

Exemple de reponse :

```json
{
  "prediction": 1,
  "label": "churn",
  "confidence": 0.78
}
```

## Prediction batch

La route `/predict_batch` accepte jusqu'a 100 entrees.

```bash
curl -X POST http://127.0.0.1:8001/predict_batch ^
  -H "Content-Type: application/json" ^
  -H "X-API-Token: Votre_clé" ^
  -d "{\"inputs\":[{\"tenure_months\":12,\"monthly_charges\":75.5,\"total_charges\":906.0,\"contract\":\"Month-to-month\"},{\"tenure_months\":24,\"monthly_charges\":90.0,\"total_charges\":2160.0,\"contract\":\"One year\"}]}"
```

Exemple de reponse :

```json
{
  "predictions": [
    {
      "prediction": 1,
      "label": "churn",
      "confidence": 0.78
    },
    {
      "prediction": 0,
      "label": "no_churn",
      "confidence": 0.65
    }
  ],
  "n_inputs": 2
}
```

Si le batch contient plus de 100 entrees, l'API retourne HTTP `413`.

Si une entree est invalide, toute la requete echoue avec HTTP `422` et l'index de l'entree invalide.

## Metriques

```bash
curl http://127.0.0.1:8001/metrics ^
  -H "X-API-Token: Votre_clé"
```

Exemple de reponse :

```json
{
  "n_predictions": 2,
  "n_errors": 0,
  "n_batch_requests": 1,
  "n_batch_inputs_total": 2
}
```

## Logs

Les logs sont visibles dans le terminal qui execute `uvicorn`.

Un batch accepte produit un log `INFO` :

```text
INFO:churn-api:batch_request_processed n_inputs=10
```

Un batch trop volumineux produit un log `WARNING` :

```text
WARNING:churn-api:batch_rejected_size=101 max_size=100
```

## Tests

### Tests automatises pytest

Lancer toute la suite de tests :

```bash
.\env\Scripts\python.exe -m pytest
```

Lancer seulement les tests API :

```bash
.\env\Scripts\python.exe -m pytest tests\test_api.py
```

Lancer les tests avec affichage detaille :

```bash
.\env\Scripts\python.exe -m pytest -v
```

Lancer seulement les tests de la route batch :

```bash
.\env\Scripts\python.exe -m pytest tests\test_api.py -k predict_batch
```

Lancer seulement les tests du pipeline de preparation :

```bash
.\env\Scripts\python.exe -m pytest tests\test_prepare.py
```

Lancer seulement les tests de non-regression du modele :

```bash
.\env\Scripts\python.exe -m pytest tests\test_non_regression.py
```

Verifier que les fichiers Python compilent :

```bash
.\env\Scripts\python.exe -m py_compile app.py main.py src\prepare.py src\train.py src\evaluate.py src\save.py tests\test_api.py
```

### Tests couverts par `tests/test_api.py`

- `GET /health` retourne `200` quand le modele et les colonnes sont charges ;
- `GET /health` retourne `503` si le modele est absent ;
- `GET /health` retourne `503` si les colonnes de features sont absentes ;
- `POST /predict` retourne une prediction avec un token valide ;
- `POST /predict` retourne `401` sans token ;
- `POST /predict` retourne `422` si un champ obligatoire manque ;
- `GET /metrics` retourne `401` avec un mauvais token ;
- `GET /metrics` expose `n_batch_requests` et `n_batch_inputs_total` ;
- `POST /predict_batch` accepte un batch valide de 2 entrees ;
- `POST /predict_batch` rejette un batch de 101 entrees avec HTTP `413` ;
- `POST /predict_batch` rejette une entree invalide avec l'index de l'entree concernee.

### Tests manuels API

Avant les tests manuels, lancer l'API :

```bash
uvicorn app:app --reload --port 8001
```

Verifier le healthcheck :

```bash
curl http://127.0.0.1:8001/health
```

Verifier que `/predict` refuse une requete sans token :

```bash
curl -X POST http://127.0.0.1:8001/predict
```

Verifier une prediction avec token :

```bash
curl -X POST http://127.0.0.1:8001/predict ^
  -H "Content-Type: application/json" ^
  -H "X-API-Token: Votre_clé" ^
  -d "{\"tenure_months\":12,\"monthly_charges\":75.5,\"total_charges\":906.0,\"contract\":\"Month-to-month\"}"
```

Verifier une prediction batch valide :

```bash
curl -X POST http://127.0.0.1:8001/predict_batch ^
  -H "Content-Type: application/json" ^
  -H "X-API-Token: Votre_clé" ^
  -d "{\"inputs\":[{\"tenure_months\":12,\"monthly_charges\":75.5,\"total_charges\":906.0,\"contract\":\"Month-to-month\"},{\"tenure_months\":24,\"monthly_charges\":90.0,\"total_charges\":2160.0,\"contract\":\"One year\"}]}"
```

Verifier les metriques :

```bash
curl http://127.0.0.1:8001/metrics ^
  -H "X-API-Token: Votre_clé"
```

Verifier le rejet d'un batch trop volumineux depuis `cmd.exe` :

```cmd
powershell -NoProfile -Command "$body = @{ inputs = @(1..101 | ForEach-Object { @{ tenure_months = 12; monthly_charges = 75.5; total_charges = 906.0; contract = 'Month-to-month' } }) } | ConvertTo-Json -Depth 5; Set-Content -Path batch_101.json -Value $body -Encoding UTF8"
curl.exe -X POST http://127.0.0.1:8001/predict_batch -H "Content-Type: application/json" -H "X-API-Token: Votre_clé" --data-binary @batch_101.json
```

La reponse attendue est HTTP `413` :

```json
{
  "detail": "Batch size is limited to 100 inputs"
}
```

Le terminal qui execute `uvicorn` doit afficher un log `WARNING` :

```text
WARNING:churn-api:batch_rejected_size=101 max_size=100
```

## Fichiers ignores

Les fichiers suivants ne doivent pas etre versionnes :

- `.env`
- `mlruns/`
- `mlflow.db`
- `artifacts/model.pkl`

Les artefacts generes peuvent etre recrees avec :

```bash
python main.py
```
