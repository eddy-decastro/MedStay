// Variables de l'infrastructure AWS documentee (jamais appliquee).

variable "aws_region" {
  description = "Region AWS. eu-west-3 = Paris, la plus proche pour la latence."
  type        = string
  default     = "eu-west-3"
}

variable "service_name" {
  description = "Nom du service, reutilise pour le depot ECR et le role IAM."
  type        = string
  default     = "medstay-ci"
}
