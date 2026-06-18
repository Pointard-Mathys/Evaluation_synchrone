import hashlib
import json
import os
import joblib


def _sha256_of_file(path: str) -> str:
    """Calcule le hash SHA-256 d'un fichier et le retourne en hexadécimal."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifacts(model, metrics: dict, feature_columns, output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    checksum_path = os.path.join(output_dir, "model.pkl.sha256")
    metrics_path = os.path.join(output_dir, "metrics.json")
    columns_path = os.path.join(output_dir, "feature_columns.json")

    joblib.dump(model, model_path)

    # Calcul et stockage du checksum SHA-256 du modèle sauvegardé
    checksum = _sha256_of_file(model_path)
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(checksum)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_columns), f, indent=2)

    if not os.path.exists(model_path):
        raise RuntimeError("Le modèle n'a pas été sauvegardé.")

    return {
        "model_path": model_path,
        "checksum_path": checksum_path,
        "metrics_path": metrics_path,
        "columns_path": columns_path,
    }