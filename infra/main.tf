// Infrastructure AWS pour MedStay-CI : ECR + App Runner.
//
// ATTENTION : ce fichier n'a JAMAIS ete applique. Voir infra/README.md pour le
// raisonnement. Le free tier AWS post-juillet 2025 facture apres 200 $ de
// credits, ce qui est incompatible avec la contrainte 0 EUR du projet.
//
// Ce code est fourni pour montrer la maitrise de l'outil, pas pour etre lance.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

// ---------------------------------------------------------------------------
// ECR : registre d'images Docker prive
// ---------------------------------------------------------------------------

resource "aws_ecr_repository" "medstay" {
  name = var.service_name

  // Analyse automatique des vulnerabilites a chaque push d'image.
  image_scanning_configuration {
    scan_on_push = true
  }

  // MUTABLE permet de reecrire le tag "latest". IMMUTABLE serait plus
  // rigoureux en production (chaque version obtient un tag unique et
  // intouchable), mais complique le flux de deploiement continu.
  image_tag_mutability = "MUTABLE"

  tags = local.tags
}

// Sans cette politique, les anciennes images s'accumulent indefiniment et le
// stockage ECR est facture au Go. On ne garde que les 5 dernieres.
resource "aws_ecr_lifecycle_policy" "conserver_5_images" {
  repository = aws_ecr_repository.medstay.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Ne conserver que les 5 images les plus recentes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

// ---------------------------------------------------------------------------
// IAM : App Runner doit pouvoir lire dans ECR, et RIEN d'autre
// ---------------------------------------------------------------------------

resource "aws_iam_role" "apprunner_ecr_access" {
  name = "${var.service_name}-ecr-access"

  // Politique de confiance : seul le service App Runner peut endosser ce role.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

// Politique geree par AWS, strictement limitee a la lecture d'ECR.
// Principe du moindre privilege : pas de droits d'ecriture, pas d'acces aux
// autres services.
resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

// ---------------------------------------------------------------------------
// App Runner : execution du conteneur
// ---------------------------------------------------------------------------

resource "aws_apprunner_service" "medstay" {
  service_name = var.service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }

    // Redeploie automatiquement des qu'une nouvelle image est poussee dans
    // ECR : equivalent de l'auto-deploy natif de Render.
    auto_deployments_enabled = true

    image_repository {
      image_identifier      = "${aws_ecr_repository.medstay.repository_url}:latest"
      image_repository_type = "ECR"

      image_configuration {
        // App Runner impose un port FIXE, contrairement a Render qui injecte
        // $PORT. C'est la seule difference reelle avec notre Dockerfile :
        // Dockerfile.aws fige 8080.
        port = "8080"
      }
    }
  }

  instance_configuration {
    // 1 vCPU / 2 Go : la plus petite taille disponible. Largement suffisante,
    // le modele LightGBM tenant en 3,4 Mo.
    cpu    = "1024"
    memory = "2048"
  }

  // Sonde de sante interrogeant l'API. App Runner retire automatiquement de la
  // rotation une instance qui ne repond plus.
  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = local.tags
}

locals {
  tags = {
    Project   = "MedStay-CI"
    ManagedBy = "Terraform"
    // Marqueur explicite : ce code n'a jamais ete applique.
    Status = "documented-not-deployed"
  }
}

// ---------------------------------------------------------------------------
// Sorties
// ---------------------------------------------------------------------------

output "ecr_repository_url" {
  description = "URL du registre ou pousser l'image"
  value       = aws_ecr_repository.medstay.repository_url
}

output "service_url" {
  description = "URL publique HTTPS du service"
  value       = "https://${aws_apprunner_service.medstay.service_url}"
}
