# Audit MLOps du projet churn

## Défaut 1 - Jeton API codé en dur et authentification absente

**Localisation** : `app.py`, ligne 12 ; routes `/predict` et `/metrics`, lignes 55 à 61

**Description** : un jeton statique est présent dans le code (`API_TOKEN = "churn-demo-token"`), mais il n'est jamais utilisé pour protéger les endpoints. Les routes de prédiction et de métriques sont donc exposées sans authentification.

**Criticité** : HAUTE

**Justification** : c'est un défaut de sécurité et de confidentialité. Un utilisateur non autorisé peut interroger le modèle, consommer les ressources de l'API et consulter les compteurs d'utilisation. Le secret codé en dur est aussi difficile à révoquer proprement et risque d'être partagé avec le dépôt.

## Défaut 2 - Chargement des artefacts sans contrôle d'intégrité

**Localisation** : `app.py`, lignes 27 à 30 ; `src/save.py`, lignes 13 à 19

**Description** : l'API charge directement `artifacts/model.pkl` avec `joblib.load`, et le pipeline sauvegarde le modèle sans signature, checksum, version de schéma ou validation d'origine.

**Criticité** : HAUTE

**Justification** : c'est un risque d'intégrité et de sécurité. Un fichier pickle modifié ou remplacé peut être chargé au démarrage de l'API. Les formats pickle/joblib ne doivent pas être considérés comme sûrs si l'artefact n'est pas maîtrisé et vérifié.

## Défaut 3 - L'API peut démarrer en état dégradé sans échouer clairement

**Localisation** : `app.py`, lignes 27 à 32 ; endpoint `/health`, lignes 50 à 52

**Description** : si les artefacts sont absents ou invalides, l'exception est seulement journalisée. L'application continue de démarrer avec `model = None`. En plus, `/health` retourne toujours un statut HTTP 200 avec `"status": "healthy"`, même si le modèle n'est pas chargé.

**Criticité** : HAUTE

**Justification** : c'est un défaut de disponibilité et de fiabilité. Un orchestrateur ou une supervision peut considérer le service comme sain alors que `/predict` retournera des erreurs 500. Cela retarde la détection d'incident et peut laisser une version inutilisable en production.

## Défaut 4 - Gestion d'erreur de prédiction trop bavarde

**Localisation** : `app.py`, lignes 82 à 85

**Description** : en cas d'erreur pendant la prédiction, l'API renvoie `detail=str(e)` au client.

**Criticité** : MOYENNE

**Justification** : cela peut exposer des détails internes comme les noms de colonnes, chemins de fichiers, erreurs de librairies ou informations de structure du modèle. Pour une API publique, il faudrait renvoyer un message générique et conserver les détails dans les logs.

## Défaut 5 - Prétraitement entraînement/API fragile et implicite

**Localisation** : `src/prepare.py`, ligne 17 ; `app.py`, lignes 67 à 69

**Description** : le preprocessing est reconstruit manuellement dans deux endroits avec `pd.get_dummies`. Le pipeline d'entraînement ne sauvegarde pas un transformateur complet, seulement la liste des colonnes. L'API réaligne ensuite les colonnes avec des zéros.

**Criticité** : MOYENNE

**Justification** : c'est un défaut de maintenabilité et de fiabilité ML. Une différence de logique entre entraînement et inférence peut créer du training-serving skew. Le modèle devrait idéalement être sauvegardé avec son préprocesseur dans un pipeline scikit-learn unique.

## Défaut 6 - Découpage des données non stratifié

**Localisation** : `src/prepare.py`, ligne 19

**Description** : `train_test_split` est appelé sans `stratify=y`.

**Criticité** : MOYENNE

**Justification** : pour un problème de churn, la classe positive peut être minoritaire. Sans stratification, le jeu de test peut ne pas représenter correctement la distribution des classes, ce qui rend les métriques instables ou trompeuses.

## Défaut 7 - Tests insuffisants et parfois peu discriminants

**Localisation** : `tests/test_prepare.py`, lignes 20 à 22 ; `tests/test_api.py`, lignes 18 à 22 et 25 à 49 ; `tests/test_non_regression.py`, lignes 11 à 19

**Description** : le test de préparation vérifie seulement que quatre objets sont retournés, sans contrôler les colonnes, les types ou la cohérence du split. Le test de santé ne vérifie pas le cas où le modèle est absent. Le test de prédiction utilise un modèle factice et ne teste pas les erreurs métier, les valeurs limites ou l'authentification.

**Criticité** : MOYENNE

**Justification** : les tests peuvent passer alors que des régressions importantes existent. Ils donnent une confiance partielle sur le fonctionnement réel du pipeline et de l'API.

## Défaut 8 - Dépendances non figées

**Localisation** : `requirements.txt`, lignes 1 à 9

**Description** : les dépendances sont listées sans versions (`pandas`, `scikit-learn`, `mlflow`, etc.).

**Criticité** : MOYENNE

**Justification** : c'est un défaut de reproductibilité et de disponibilité. Une nouvelle version d'une librairie peut casser l'entraînement, les tests, l'API ou changer les résultats du modèle. En MLOps, l'environnement doit être reproductible.

## Défaut 9 - CI minimale sans contrôles qualité ni sécurité

**Localisation** : `.github/workflows/ci.yml`, lignes 20 à 29

**Description** : le workflow installe les dépendances, exécute le pipeline puis les tests, mais ne lance pas de lint, formatage, analyse de sécurité, contrôle de couverture, validation d'artefacts ou publication contrôlée du modèle.

**Criticité** : BASSE

**Justification** : la CI détecte seulement une partie des erreurs fonctionnelles. Des problèmes de style, de secrets, de dépendances vulnérables, de couverture insuffisante ou d'artefacts invalides peuvent passer dans la branche principale.

## Défaut 10 - Documentation d'exploitation incomplète

**Localisation** : `README.md`, lignes 15 à 33

**Description** : le README indique comment lancer le pipeline et l'API, mais ne documente pas le format exact d'entrée de `/predict`, les variables de configuration, la gestion des artefacts, la procédure de monitoring, les limites connues ou les étapes de déploiement.

**Criticité** : BASSE

**Justification** : c'est un défaut de maintenabilité et d'exploitation. Un nouveau développeur ou opérateur peut lancer le projet, mais il manque les informations nécessaires pour diagnostiquer, reproduire et opérer correctement le service.
