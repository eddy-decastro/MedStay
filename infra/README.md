# `infra/` — Déploiement AWS documenté, volontairement non exécuté

## Pourquoi ce dossier existe et pourquoi rien n'y tourne

Ce dossier contient une infrastructure AWS complète et fonctionnelle sur le
papier (Terraform + Dockerfile App Runner). **Elle n'a jamais été appliquée, et
c'est un choix documenté, pas une lacune.**

Le projet est soumis à une contrainte non négociable : **0 € de coût, aucune
carte bancaire**. Or depuis juillet 2025, le free tier AWS a changé de nature :
il offre 200 $ de crédits sur 6 mois, après quoi la facturation démarre. Un
`terraform apply` oublié sur App Runner coûte environ 5 $ par mois — pas
ruineux, mais incompatible avec la contrainte posée.

Le raisonnement d'ingénieur est le suivant : **savoir déployer sur AWS et
choisir de ne pas le faire vaut mieux que déployer sans comprendre la
facturation.** Le code est ici, commenté, prêt à être appliqué par qui accepte
d'en assumer le coût.

## Ce que ferait cette infrastructure

```
GitHub  ──push──►  ECR (registre d'images)
                        │
                        ▼
                   App Runner  ──►  service HTTPS public, mise à l'échelle auto
```

- **ECR** : héberge l'image Docker (équivalent d'un Docker Hub privé).
- **App Runner** : exécute le conteneur, gère HTTPS, le domaine et la montée en
  charge. C'est l'équivalent AWS le plus proche de Render.
- **IAM** : rôle minimal autorisant App Runner à lire dans ECR, rien de plus.

## Fichiers

| Fichier | Rôle |
|---|---|
| `main.tf` | ECR + App Runner + rôle IAM, entièrement commenté |
| `variables.tf` | Région, nom du service, taille d'instance |
| `Dockerfile.aws` | Variante du Dockerfile pour App Runner (port fixe 8080) |

## Ce qui change par rapport à Render

| | Render (utilisé) | AWS App Runner (documenté) |
|---|---|---|
| Port | `$PORT` dynamique injecté | 8080 fixe, déclaré dans la config |
| Déploiement | webhook GitHub natif | push d'image vers ECR |
| Coût | 0 € | ~5 $/mois après les crédits |
| Mise en veille | après 15 min | aucune |
| Infrastructure | 3 clics dans l'interface | ~80 lignes de Terraform |

## Comment l'appliquer, si on l'assume

```bash
cd infra
terraform init
terraform plan     # à lire ATTENTIVEMENT : cette étape est gratuite
terraform apply    # à partir d'ici, la facturation démarre
terraform destroy  # à ne pas oublier
```

Ne jamais lancer ces commandes en CI : ce dossier est exclu du workflow
GitHub Actions précisément pour éviter un déploiement accidentel.
